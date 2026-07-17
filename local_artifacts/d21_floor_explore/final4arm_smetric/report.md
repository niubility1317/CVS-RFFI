# D21 floor-aware极轻量数学描述符开发实验

状态：`DEVELOPMENT_QUERY_DIAGNOSTIC_COMPLETED`。本实验仅覆盖rx20-1、开发seed713101、K10、5个真实新TX与三个互斥`LEO_weak`场景，不是独立确认矩阵，也不构成目标达成声明。

## 实验边界

- 输入：既有封存K10/new5 Phase2 capsule；predictor只读取注册support、无truth的LEO query及冻结runtime。
- truth仅由predict完成后运行的独立score命令连接；所有表征权重、metric配置与blend系数均由三场景support-LOO联合锁定。
- 每个物理样本仅使用capsule中的固定LEO_weak接收IQ；DP/RF/FFT均为同一IQ上的确定性计算，不生成子样本，不改变K，每个样本最多计算一次FFT96。
- query逐样本面对全部11个已注册类；无query拟合、角色Oracle、类别quota、全局重排或dense query图。
- 4个descriptor arm严格固定为：A0=`z+FFT96`，A1=`z+FFT96+RF32`，A2=`z+FFT96+DP16/32`，A3=`z+FFT96+RF32+DP16/32`。DP仅由相邻包裹差分相位的圆均值、合向量长度、稳健尺度及固定直方图组成。

## 4个descriptor arm结果

| arm | support锁定配置 | 维数 | old before | old after | old floor | seen-new | new floor | H | forgetting |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 z+FFT96 | FFT=8 | 256 | 0.8306 | 0.7278 | 0.5167 | 0.7667 | 0.6167 | 0.7467 | 0.1028 |
| A1 z+FFT96+RF32 | FFT=4,RF=4 | 288 | 0.8222 | 0.7306 | 0.5000 | 0.7233 | 0.6000 | 0.7269 | 0.0917 |
| A2 z+FFT96+DP32 | FFT=4,DP=4 | 288 | 0.7917 | 0.6611 | 0.4167 | 0.7533 | 0.5833 | 0.7042 | 0.1306 |
| A3 z+FFT96+RF32+DP16 | FFT=4,RF=2,DP=2 | 304 | 0.8389 | 0.7306 | 0.4833 | 0.7367 | 0.5667 | 0.7336 | 0.1083 |

support-LOO按worst-scene old floor优先选择A2，但A2在query上明显弱于A0，说明该开发切片存在support→query表征排序失配，A2不得晋升。

## 三场景floor

| arm | scenario | old after | old floor | seen-new | new floor | H | forgetting |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | clear | 0.8417 | 0.7000 | 0.8700 | 0.7000 | 0.8556 | 0.0750 |
| A0 | low-elev | 0.6583 | 0.4000 | 0.7400 | 0.6000 | 0.6968 | 0.0917 |
| A0 | rain | 0.6833 | 0.4000 | 0.6900 | 0.5500 | 0.6867 | 0.1417 |
| A1 | clear | 0.7583 | 0.6500 | 0.8200 | 0.6000 | 0.7880 | 0.1250 |
| A1 | low-elev | 0.6833 | 0.2500 | 0.6600 | 0.4500 | 0.6715 | 0.0917 |
| A1 | rain | 0.7500 | 0.5000 | 0.6900 | 0.6000 | 0.7188 | 0.0583 |
| A2 | clear | 0.6917 | 0.5500 | 0.8000 | 0.6000 | 0.7419 | 0.1417 |
| A2 | low-elev | 0.6500 | 0.2500 | 0.7600 | 0.5500 | 0.7007 | 0.1417 |
| A2 | rain | 0.6417 | 0.3500 | 0.7000 | 0.5000 | 0.6696 | 0.1083 |
| A3 | clear | 0.7750 | 0.6500 | 0.8100 | 0.6000 | 0.7921 | 0.1250 |
| A3 | low-elev | 0.7000 | 0.2500 | 0.7100 | 0.4500 | 0.7050 | 0.1250 |
| A3 | rain | 0.7167 | 0.5000 | 0.6900 | 0.5500 | 0.7031 | 0.0750 |

low-elev是所有固定concat的共同旧类floor瓶颈。RF32改善rain old floor但破坏low-elev和新类均值；DP没有形成可外推的floor增益。

## Floor-aware diagonal metric

对固定A0 256维表征，before使用old support拟合`theta_B`，after以`theta_B`初始化`theta_C`，损失包含support-LOO加权CE、top-30%类CVaR、old-support logit蒸馏、old pairwise preservation、新类侵入hinge与对角权重正则。三场景support-LOO锁定`L6_diag_floor`与`alpha=1.0`。

| candidate | params | epoch | old before | old after | old floor | seen-new | new floor | H | forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 fixed | 0 | 0 | 0.8306 | 0.7278 | 0.5167 | 0.7667 | 0.6167 | 0.7467 | 0.1028 |
| D256 support-selected | 256 | 20 | 0.8861 | 0.7500 | 0.6000 | 0.8200 | 0.5667 | 0.7834 | 0.1361 |
| D256 strong-distill diagnostic | 256 | 20 | 0.8722 | 0.7583 | 0.5833 | 0.8067 | 0.5667 | 0.7818 | 0.1139 |
| A2 selected descriptor+sequential metric | 288 | 20 | 0.8500 | 0.7500 | 0.5167 | 0.7867 | 0.6167 | 0.7679 | 0.1000 |

D256提高old、新类与H并把聚合old floor提高8.33pp，但遗忘恶化3.33pp。strong distill减轻遗忘但未被预登记support排序选中。A2上的sequential metric由support锁定为`floor_distill,alpha=0.25`，仍不如A0 D256，进一步证明A2的support优势不能外推query。

## 完整loss trace核验

- `loss_trace.jsonl`共720条，36个candidate×scenario×phase组；每组严格包含epoch1–20，无缺失、重复或非有限值。
- A0 metric的after-registration loss均下降：clear的support joint floor最高0.70，low-elev最高0.40，rain最高0.50。
- selected A2 metric的support-LOO改进未转化为query floor，根因更接近support/query局部邻域失配，而不是优化未收敛或NaN/collapse。

## 资源审计

| candidate | trainable params | epoch | persistent state | query classifier MAC | classifier mean/P95 ms |
|---|---:|---:|---:|---:|---:|
| A0 | 0 | 0 | 28,380B | 28,160 | 0.00124/0.00175 |
| A1 | 0 | 0 | 31,900B | 31,680 | 0.00123/0.00169 |
| A2 | 0 | 0 | 31,900B | 31,680 | 0.00121/0.00185 |
| A3 | 0 | 0 | 33,660B | 33,440 | 0.00120/0.00167 |
| D256 | 256 | 20 | 29,404B | 29,184 | 0.00264/0.00384 |
| A2 sequential metric | 288 | 20 | 33,056B | 64,512 | 0.00344/0.00645 |

support码实际采用逐向量int8＋FP16 scale；optimizer状态不持久化。D256单次after适配估算61,952,000MAC，A2 sequential约69,696,000MAC。实测峰值CUDA显存64,990,208B。全部候选低于50k参数、20epoch与256KB持久状态上限。

## Artifact与复现

| artifact | SHA256 |
|---|---|
| predictions_k10_new5.npz | `9c75791029bd1961c4d02229740d29f2c9eb0cb97a575107e633c0b3ec06172d` |
| predictions_k10_new5.receipt.json | `3ada3a0334538e1bd931360b45262eee5be09438d72760b4c84c75c0894e78be` |
| loss_trace.jsonl | `6f655d65632bf4ef7f383b0a120af3e14c4ccdd0a4d69175b6eecb8247c5098a` |
| score_k10_new5.json | `d3ce46abd1376bad685b7b8345e55408e95c5dd476fa344c379d80d16b1dc153` |

```powershell
conda run -n ssr-gpu python local_artifacts/d21_floor_explore/run_floor_aware_diag.py predict --capsule E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5 --output local_artifacts\d21_floor_explore\final4arm_smetric\predictions_k10_new5.npz
conda run -n ssr-gpu python local_artifacts/d21_floor_explore/run_floor_aware_diag.py score --prediction local_artifacts\d21_floor_explore\final4arm_smetric\predictions_k10_new5.npz --truth E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\scorer\truth_sidecar.json --output local_artifacts\d21_floor_explore\final4arm_smetric\score_k10_new5.json
```

## 结论

本轮最有效路线仍是A0的256参数floor-aware diagonal metric；RF/DP固定concat未解决low-elev旧类floor，DP还暴露support→query排序失配。开发结果距离`old_acc>=0.92`、`min_old_class_acc>=0.88`、new5`seen_new_acc>=0.92`仍很远，不能进入125确认矩阵。下一轮若继续，应优先修复support-LOO代理目标与query局部邻域的一致性，而不是继续增加描述符或logit bias。
