#pragma once

// 对长度为 n 的设备端 float 数组求和，结果写入 d_out[0]
void reduce_sum(const float* d_in, float* d_out, int n);
