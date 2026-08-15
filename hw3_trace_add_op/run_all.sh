#!/usr/bin/env bash
# 一键跑完全部脚本，输出保存到 logs/ 供批改/回顾。
# 从 hw3_trace_add_op 目录下执行： bash run_all.sh
set -eu

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="${HERE}/logs"
mkdir -p "${LOG_DIR}"

for f in 01_dump_dispatch_table.py 02_torch_dispatch_trace.py 03_profiler_trace.py 04_logging_and_hooks.py; do
    echo ">>> running ${f}"
    log="${LOG_DIR}/${f%.py}.log"
    python3 "${HERE}/${f}" > "${log}" 2>&1 || { echo "  FAILED, see ${log}"; }
    echo "    -> ${log}  ($(wc -l < "${log}") lines)"
done

echo
echo "All done. Chrome traces:"
ls -1 "${HERE}"/*.json 2>/dev/null || true
