# Lesson 03 练习结论 — GPU Setup & Cloud

本机：NVIDIA GeForce RTX 4050 Laptop GPU，VRAM ≈ 6.4 GB（nvidia-smi 约 6141 MiB）

## 1. CPU vs GPU benchmark

脚本：`GPU_vs_CPU_Benchmark.py`（5000×5000 矩阵乘）

| 设备 | 耗时 |
|------|------|
| CPU  | 0.406s |
| GPU  | 0.121s |
| 加速比 | ≈ 3.4×（约 3×） |

结论：同规模稠密矩阵乘上，本地 4050 相对 CPU 有明显加速；笔记本 GPU 带宽/功耗受限，加速比通常小于桌面 4090。

## 2. 无本地 GPU 时

本机有 CUDA（`torch.cuda.is_available() == True`），未使用 Colab。  
无 GPU 时可按课文走 Google Colab（Runtime → T4 GPU）跑同一段 benchmark。

## 3. 最大可装模型（fp16 拇指法则）

规则：fp16 ≈ **2 bytes / 参数**

\[
\text{最大参数量} \approx \frac{\text{VRAM}}{2}
\]

| 卡 | VRAM | fp16 权重上限（粗估） | 推理留余量后更稳妥 |
|----|------|----------------------|-------------------|
| 本机 RTX 4050 | ~6 GB | ~3B（约 30 亿） | ~1B–2B |
| 对照 RTX 4090 | 24 GB | ~12B | ~7B–10B |

说明：上表只计**权重**。训练还需优化器状态、激活等，可训规模通常远小于推理上限。
