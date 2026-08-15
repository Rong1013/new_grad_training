import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, record_function
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent


class SlowPatternDataset(Dataset):
    """Synthetic image dataset with intentionally slow CPU-side preprocessing."""

    def __init__(
        self,
        samples: int,
        num_classes: int,
        image_size: int,
        cpu_aug_rounds: int,
        data_sleep_ms: float,
    ) -> None:
        self.samples = samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.cpu_aug_rounds = cpu_aug_rounds
        self.data_sleep_ms = data_sleep_ms / 1000.0

        generator = torch.Generator().manual_seed(2026)
        self.templates = torch.randn(
            num_classes, 3, image_size, image_size, generator=generator
        ) * 0.03
        patch = max(4, image_size // 6)
        for label in range(num_classes):
            channel = label % 3
            row = (label * 5) % (image_size - patch)
            col = (label * 7) % (image_size - patch)
            self.templates[label, channel, row : row + patch, col : col + patch] += 1.0

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = index % self.num_classes
        generator = torch.Generator().manual_seed(index)
        image = self.templates[label] + 0.08 * torch.randn(
            3, self.image_size, self.image_size, generator=generator
        )

        # Deliberately CPU-heavy preprocessing to make DataLoader/CPU bottlenecks visible.
        for _ in range(self.cpu_aug_rounds):
            image = image.flip(-1).contiguous().flip(-1)
            image = (image * 1.001).clamp(-3.0, 3.0)

        if self.data_sleep_ms > 0:
            time.sleep(self.data_sleep_ms)

        return image.float(), torch.tensor(label, dtype=torch.long)


class BottleneckNet(nn.Module):
    """Small CNN/MLP with labeled performance bottleneck regions."""

    def __init__(
        self,
        num_classes: int,
        hidden: int,
        fragmented_chunks: int,
        alloc_rounds: int,
    ) -> None:
        super().__init__()
        if hidden % fragmented_chunks != 0:
            raise ValueError("--hidden must be divisible by --fragmented-chunks")

        self.fragmented_chunks = fragmented_chunks
        self.alloc_rounds = alloc_rounds
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(16 * 4 * 4, hidden)
        self.slow_weight = nn.Parameter(torch.randn(hidden, hidden) * 0.02)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = torch.flatten(x, 1)
        x = torch.relu(self.proj(x))

        with record_function("bottleneck_single_large_matmul"):
            x = torch.relu(x @ self.slow_weight)

        with record_function("bottleneck_fragmented_small_ops"):
            parts = []
            for chunk_id, chunk in enumerate(torch.chunk(x, self.fragmented_chunks, dim=1)):
                part = chunk + (chunk_id + 1) * 1e-4
                part = torch.relu(part)
                part = torch.sigmoid(part) * part
                part = part.clone()
                parts.append(part)
            x = torch.cat(parts, dim=1)

        with record_function("bottleneck_frequent_allocations"):
            for _ in range(self.alloc_rounds):
                detached_scratch = torch.empty_like(x)
                detached_scratch.copy_(x.detach())
                grad_scratch = x.clone()
                grad_scratch = grad_scratch.transpose(0, 1).contiguous()
                grad_scratch = grad_scratch.transpose(0, 1).contiguous()
                x = x + grad_scratch * 1e-4 + detached_scratch * 1e-5

        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small PyTorch model with intentional profiler bottlenecks."
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--fragmented-chunks", type=int, default=16)
    parser.add_argument("--alloc-rounds", type=int, default=2)
    parser.add_argument("--cpu-aug-rounds", type=int, default=2)
    parser.add_argument("--data-sleep-ms", type=float, default=0.2)
    parser.add_argument("--python-work", type=int, default=3000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "profiler_traces")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(name)


def build_profiler(args: argparse.Namespace, device: torch.device):
    if not args.profile:
        return nullcontext(None)

    from torch.profiler import profile

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    return profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    profiler,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for step, (images, labels) in enumerate(dataloader, start=1):
        with record_function("bottleneck_h2d_copy_batch"):
            images = images.to(device)
            labels = labels.to(device)

        with record_function("bottleneck_cpu_python_work"):
            checksum = 0
            for i in range(args.python_work):
                checksum += (i * i) % 97

        with record_function("bottleneck_extra_memcpy_and_sync"):
            cpu_noise = torch.randn(images.size(0), 1, 1, 1)
            images = images + cpu_noise.to(device) * 1e-4

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)

        with record_function("bottleneck_loss_temp_allocations"):
            one_hot = torch.zeros(
                labels.size(0), logits.size(1), device=device, dtype=logits.dtype
            )
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            loss = loss + 0.02 * (torch.softmax(logits, dim=1) - one_hot).pow(2).mean()

        loss.backward()
        optimizer.step()

        with record_function("bottleneck_d2h_sync_for_logging"):
            predictions_cpu = logits.detach().argmax(dim=1).cpu()
            labels_cpu = labels.detach().cpu()
            batch_correct = int((predictions_cpu == labels_cpu).sum().item())
            batch_loss = float(loss.detach().cpu())

        total_loss += batch_loss * labels.size(0)
        total_correct += batch_correct
        total_seen += labels.size(0)

        if profiler is not None:
            profiler.step()

        if step == 1 or step == len(dataloader):
            print(
                f"epoch={epoch} step={step}/{len(dataloader)} "
                f"loss={batch_loss:.4f} acc={batch_correct / labels.size(0):.3f} "
                f"checksum={checksum}"
            )

    return total_loss / total_seen, total_correct / total_seen


def print_profiler_report(profiler, args: argparse.Namespace, device: torch.device) -> None:
    if profiler is None:
        return

    time_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    memory_key = (
        "self_cuda_memory_usage" if device.type == "cuda" else "self_cpu_memory_usage"
    )
    print("\nProfiler top operators by time:")
    print(profiler.key_averages(group_by_input_shape=True).table(sort_by=time_key, row_limit=20))

    print("\nProfiler top operators by memory allocation:")
    print(profiler.key_averages().table(sort_by=memory_key, row_limit=20))

    trace_path = args.trace_dir / f"bottleneck_trace_{device.type}.json"
    profiler.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace exported to: {trace_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(2026)
    os.environ.setdefault("TORCH_HOME", str(ROOT / "torch_home"))

    device = select_device(args.device)
    print(f"device={device}")
    print(
        "bottleneck labels: "
        "h2d_copy_batch, cpu_python_work, extra_memcpy_and_sync, "
        "single_large_matmul, fragmented_small_ops, frequent_allocations, "
        "d2h_sync_for_logging"
    )

    dataset = SlowPatternDataset(
        samples=args.samples,
        num_classes=args.num_classes,
        image_size=args.image_size,
        cpu_aug_rounds=args.cpu_aug_rounds,
        data_sleep_ms=args.data_sleep_ms,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" and args.num_workers > 0),
    )

    model = BottleneckNet(
        num_classes=args.num_classes,
        hidden=args.hidden,
        fragmented_chunks=args.fragmented_chunks,
        alloc_rounds=args.alloc_rounds,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    started_at = time.perf_counter()
    with build_profiler(args, device) as prof:
        for epoch in range(1, args.epochs + 1):
            avg_loss, avg_acc = run_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                args=args,
                epoch=epoch,
                profiler=prof,
            )
            print(f"epoch={epoch} summary loss={avg_loss:.4f} acc={avg_acc:.3f}")

    elapsed = time.perf_counter() - started_at
    print(f"training_completed elapsed_sec={elapsed:.2f}")
    print_profiler_report(prof, args, device)


if __name__ == "__main__":
    main()
