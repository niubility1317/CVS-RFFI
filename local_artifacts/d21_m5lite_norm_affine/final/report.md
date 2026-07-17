# D21 M5-lite极轻norm-affine诊断

## 结论

本次固定路线为负结果，停止继续调参。M5-lite与Z0的聚合`old_acc/seen_new_acc/H_old_new/floor/forgetting`完全相同，仅改变660个after-query中的7个预测；场景级改善与退化互相抵消。support损失仅轻微下降，support准确率与floor基本不动，不构成support过拟合，更接近白名单更新幅度过小、路线无效。

该结果只属于单个formal K=10/new=5 capsule的Stage2-C诊断，不是正式多receiver、多seed确认矩阵，也未达到项目目标门槛。

## 协议与锁定

- 输入仅来自密封capsule中的单一`LEO_weak`观测：`support_leo_weak_iq`、`query_leo_weak_iq`。未读取clean/source IQ、clean/source特征、clean衍生信号或额外信道视图。
- FFT96是对已经进入Phase2的同一个`LEO_weak` IQ观测进行固定变换，不生成或引入另一份clean/信道样本。
- 适配只读取注册support IQ及support标签；query标签不进入训练、适配、校准、选择、早停、回滚或候选排名。
- 唯一M5配置在打开query truth前固定：SGD、`lr=0.01`、momentum=0、5个full-support epoch/step、support CE+类CVaR+old pairwise retention。没有超参数网格。
- Z0与固定M5-lite各生成一次全query预测；每个query独立在全部11个注册类上打分，无角色Oracle、类别配额、真实batch类数或全局分配。
- prediction NPZ先写入并计算SHA-256；独立`score`命令才打开truth sidecar。score结果没有反馈到predictor。看到负结果后停止，不再调参。
- M5属于Stage2-C only。注册前指标使用共享的Z0 old-only状态；没有用包含seen-new support的M5状态伪装“注册前”状态。

## 精确白名单与更新

仅允许`model.id_backbone.time_fuse.1.{weight,bias}`以及`t1/t2/t3/f1/f2/f3/pa_b1/pa_b2/pa_b3`的`.norm.{weight,bias}`。共20个张量、1136个可训练标量；`dom_backbone`和其余参数全部冻结。

| 场景 | 可训练参数 | FP16非零delta | epoch/step | 最大绝对delta | delta L2 |
|---|---:|---:|---:|---:|---:|
| leo_clear_weak | 1136 | 1135 | 5/5 | 0.002073 | 0.012255 |
| leo_low_elev_weak | 1136 | 1135 | 5/5 | 0.003067 | 0.015293 |
| leo_rain_weak | 1136 | 1135 | 5/5 | 0.001747 | 0.012295 |

最终预测由重新加载的密封基线加FP16 delta后生成，不使用未量化训练态。未保存optimizer state。

## 最终测试结果

| 方法 | 范围 | old-before | old-after | seen-new | old floor | new floor | H | forgetting |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Z0 | 聚合 | 83.06% | 72.78% | 76.67% | 51.67% | 61.67% | 74.67% | 10.28pp |
| M5-lite | 聚合 | 83.06% | 72.78% | 76.67% | 51.67% | 61.67% | 74.67% | 10.28pp |
| Z0 | clear | 91.67% | 84.17% | 87.00% | 70.00% | 70.00% | 85.56% | 7.50pp |
| M5-lite | clear | 91.67% | 84.17% | 86.00% | 70.00% | 70.00% | 85.07% | 7.50pp |
| Z0 | low_elev | 75.00% | 65.83% | 74.00% | 40.00% | 60.00% | 69.68% | 9.17pp |
| M5-lite | low_elev | 75.00% | 66.67% | 75.00% | 45.00% | 60.00% | 70.59% | 8.33pp |
| Z0 | rain | 82.50% | 68.33% | 69.00% | 40.00% | 55.00% | 68.67% | 14.17pp |
| M5-lite | rain | 82.50% | 67.50% | 69.00% | 35.00% | 55.00% | 68.24% | 15.00pp |

M5-lite相对Z0改变的after-query数为clear 1/220、low_elev 3/220、rain 3/220。low_elev小幅变好，但clear新类和rain旧类退化；聚合无收益，且rain旧类floor从40%降到35%。

## support轨迹与过拟合判断

| 场景 | total loss epoch1→5 | old acc epoch1→5 | new acc epoch1→5 | old floor epoch1→5 | new floor epoch1→5 |
|---|---:|---:|---:|---:|---:|
| clear | 2.62998→2.62746 | 86.67→86.67% | 92.00→92.00% | 40→40% | 60→60% |
| low_elev | 2.75143→2.74738 | 73.33→71.67% | 72.00→72.00% | 40→40% | 40→40% |
| rain | 2.77510→2.77265 | 73.33→73.33% | 70.00→70.00% | 30→30% | 40→40% |

没有出现“support显著变好、query变差”的典型过拟合；support本身几乎未改善，low_elev旧类反而下降1.67pp。路线失败主要是5步、1136参数norm-affine对固定FFT96主导表示作用太弱，而不是可利用的训练收益未泛化。

## 资源审计

| 项目 | 审计值 |
|---|---:|
| adapter可训练参数 | 1136（上限50000的2.27%） |
| FP16 delta有效payload | 2272B/场景（2.22KiB） |
| 三场景NPZ含名称/ZIP开销 | 23792B |
| 持久optimizer状态 | 0B |
| 适配epoch/step | 5/5（上限20/50） |
| 适配时间 | 2.865–3.499s/场景 |
| 峰值CUDA显存 | 234079744B（223.24MiB） |
| merge后adapter新增推理MAC | 0 |
| 固定FFT96单qKNN分类MAC | 28160/query |
| Z0/M5特征推理延迟 | clear 9.39/10.19、low 10.77/9.88、rain 11.66/13.35ms/query |
| KNN分类延迟 | 0.00149–0.00196ms/query |

M5 delta合并进已有affine层后不增加模型前向MAC；计时波动未形成稳定延迟优势。TorchScript密封运行时内部绑定CUDA，本地无法以CPU执行，此处显存是实际本地GPU峰值，不主张为星上CPU/NPU实测。

## 证据文件

- `predictions_k10_new5.npz`：不可变式预测，SHA-256 `88ff60c45c4dce43b8a92738cfde6abd1c926c1a66a93d524b4bc1daf7552fa1`
- `predictions_k10_new5.receipt.json`：白名单、资源与协议receipt
- `m5lite_fp16_delta.npz`：FP16补丁，SHA-256 `ce4be36ebd7e99219052568d783f6ccee981c076871bdc6232ed5fa999558045`
- `loss_trace.jsonl`：15条完整训练记录，SHA-256 `3b10f66af1a8e38ada218e898a96e580b69ad1d2a9fa69ca49337883e4efb13a`
- `score_k10_new5.json`：隔离scorer结果
- `../run_m5lite_norm_affine.py`：可复现predict/score实现

## 决策

`STOP_NEGATIVE_DIAGNOSTIC`。不扩大网格、不依据query调参、不进入正式矩阵或部署候选。该结论仅否定当前固定的5步norm-affine配置，不等同于否定所有模型级适配路线。
