# CVCNN vs Sinc-CVCNN 物理感知前端验证方案

日期：2026-05-08

## 1. 研究问题

目标不是再造一个强模型，而是做一个干净的因果验证：

> 在完全相同的 CVCNN 后端、训练策略、损失函数和数据划分下，仅把第一层自由复卷积替换为 Hz 参数化 SincNet 滤波器，是否能提升未见接收机、未见日期、星地信道 overlay 和 worst-RX 性能？

如果成立，说明“物理感知第一层滤波器组”确实提供了可泛化归纳偏置；如果不成立，则主模型的收益更多来自后续多分支解耦、MixStyle/Fishr、PA 辅助或训练策略。

## 2. 网络结构

### 2.1 普通 CVCNN

输入：`x in R^[B,2,L]`，两通道为 I/Q。

结构：

1. `ComplexBlock(1 -> C, k=7, pool=2)`：自由复卷积第一层。
2. `ComplexBlock(C -> 2C, k=5, pool=2)`。
3. `ComplexBlock(2C -> 4C, k=3, pool=2)`。
4. `AdaptiveAvgPool1d(1)`。
5. `Linear(8C -> embedding_dim)`。
6. `Linear(embedding_dim -> num_tx)`。

损失：只用 transmitter cross entropy。

### 2.2 Sinc-CVCNN

除第一层外全部相同：

1. `SincConv1d(C, k=79)` 分别作用在 I 和 Q 上，共享同一组 Hz 参数化带通滤波器。
2. 拼接为 `2C` 通道，接 `BN + ReLU + AvgPool1d(2)`。
3. 后续 `ComplexBlock(C -> 2C)`、`ComplexBlock(2C -> 4C)`、池化、embedding、分类头全部与普通 CVCNN 相同。

设计含义：

- 普通 CVCNN 第一层学习任意时域卷积核；
- Sinc-CVCNN 第一层被限制为可解释带通滤波器组，只学习低截止频率与带宽；
- 两者从第二层开始完全一致，因此差异主要归因于第一层物理先验。

## 3. 已落地代码

新增/修改：

- `baselines/cvcnn/model.py`
  - `SincComplexStem`
  - `SincCVCNN`
  - `_CVCNNTail`，确保普通 CVCNN 和 Sinc-CVCNN 共用同一个 tail。
- `baselines/cvcnn/train_cvs.py`
  - 新增 `--front_end conv|sinc`
  - 新增 `--sinc_kernel_size`
  - 新增 `--sample_rate_hz`
- `run_cvcnn_sinc_frontend_queue.sh`
  - 固定 seed=1337；
  - 默认运行 `C in {24,32,48,64}` 下的 conv/sinc 成对实验；
  - 默认 `--eval_sat_on all`，星地评估覆盖所有 named split。

## 4. 实验组

固定：

- seed：1337
- loss：CE only
- optimizer：AdamW
- 数据划分：与当前 WiSig split 一致
- SAT eval：`clear_leo,low_elev_leo,rain_leo,storm_leo_mp`
- SAT split：`all`

实验：

| 组别 | front_end | C | 目的 |
|---|---|---:|---|
| `cvcnn_conv_c24` | conv | 24 | 小容量自由卷积 |
| `cvcnn_sinc_c24` | sinc | 24 | 小容量 Sinc 物理前端 |
| `cvcnn_conv_c32` | conv | 32 | 默认容量自由卷积 |
| `cvcnn_sinc_c32` | sinc | 32 | 默认容量 Sinc 物理前端 |
| `cvcnn_conv_c48` | conv | 48 | 较大容量自由卷积 |
| `cvcnn_sinc_c48` | sinc | 48 | 较大容量 Sinc 物理前端 |
| `cvcnn_conv_c64` | conv | 64 | 大容量自由卷积 |
| `cvcnn_sinc_c64` | sinc | 64 | 大容量 Sinc 物理前端 |

这样能检查 Sinc 的收益是否只是参数量/容量差异。如果 Sinc 在多个 C 下都提升 OOD 或 SAT，则证据更强。

## 5. 成功判据

Sinc 物理前端有效，需要满足至少三条：

1. 同容量下 primary OOD 不低于普通 CVCNN，最好提升 >= 0.30。
2. strict UDU 或 worst-RX 提升 >= 0.30。
3. SAT Avg 或 `all_named_tx` 提升 >= 0.80。
4. `storm_leo_mp`、`low_elev_leo` 至少一个明显提升。
5. 不出现某个核心 split 大幅下降。

如果只在 clean val 上提升、但未见 rx/SAT 不提升，则不能证明物理感知有效。

## 6. 启动命令

```bash
cd ~/2510044040/CV-SincNet
mkdir -p logs baseline_runs/cvcnn_physics
GPU_IDS=0,1,2,3,4,5,6,7 SEED=1337 EPOCHS=200 \
nohup bash run_cvcnn_sinc_frontend_queue.sh \
  > logs/cvcnn_sinc_frontend_$(date +%Y%m%d_%H%M%S).nohup.log 2>&1 &
```

快速 smoke：

```bash
GPU_IDS=0,1 EPOCHS=2 BASE_CHANNELS=24 SAT_EVAL_MAX_BATCHES=1 \
bash run_cvcnn_sinc_frontend_queue.sh
```
