# D21 M1双状态竞争保护开发实验

状态：`DEVELOPMENT_QUERY_DIAGNOSTIC_COMPLETED`。本实验仅覆盖rx20-1、开发seed713101、K10、5个真实新TX与三个互斥`LEO_weak`场景，不是独立确认矩阵或正式性能声明。

## 方法与权限边界

- 固定表征：`A0=normalize(concat(normalize(z_id160),8*normalize(FFT96)))`。
- Stage2-B只用old support拟合`theta_B`并冻结old int8 support codes；Stage2-C从`theta_B`初始化`theta_C`，只用注册support标签拟合new空间。
- 每个query独立计算全部old类的`score_B`与全部new类的`score_C`，再进行support锁定的跨空间校准；predictor不知道query真实old/new角色。
- `theta_C`损失包含new-class CE、new-class CVaR、old support侵入hinge、new support保真hinge、old pairwise preservation和`theta_C-theta_B`正则。
- 跨空间网格严格限制为`T_old,T_new∈{0.9,1.0,1.1}`及`new_offset∈{0,0.02,0.04}`，只由三场景self-excluded support联合锁定。
- support门固定要求old侵入率不超过25%、new保真率不低于45%。首次55%保真门在打开truth前阻断radius分支，因此仅依据support证据预登记为45%后从头重跑；未读取query标签调门。
- query truth只在不可变prediction生成后由独立score命令连接；无query拟合、角色Oracle、类别quota、真实batch类别数或全局重排。

## 聚合结果

| candidate | old before | old after | old floor | seen-new | new floor | H | forgetting | support gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pure L6q single metric | 0.8667 | 0.7611 | 0.5833 | 0.8167 | 0.5500 | 0.7879 | 0.1056 | n/a |
| M1 dual metric | 0.8667 | 0.8556 | 0.7500 | 0.6333 | 0.3333 | 0.7279 | 0.0111 | PASS |
| M1 dual+radius | 0.4639 | 0.2944 | 0.0333 | 0.4400 | 0.0333 | 0.3528 | 0.1694 | BLOCKED diagnostic |

raw dual把旧类遗忘降低9.44pp，并把old floor提高16.67pp，但seen-new下降18.33pp、new floor下降21.67pp。这是明确的old/new Pareto交换，不是目标达成。

## 三场景结果

| candidate | scenario | old before | old after | old floor | new | new floor | H | forgetting |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pure L6q | clear | 0.9250 | 0.8500 | 0.7000 | 0.8400 | 0.5500 | 0.8450 | 0.0750 |
| pure L6q | low-elev | 0.8167 | 0.7000 | 0.4500 | 0.7900 | 0.4000 | 0.7423 | 0.1167 |
| pure L6q | rain | 0.8583 | 0.7333 | 0.5000 | 0.8200 | 0.7000 | 0.7743 | 0.1250 |
| M1 dual | clear | 0.9250 | 0.9250 | 0.9000 | 0.5800 | 0.1000 | 0.7130 | 0.0000 |
| M1 dual | low-elev | 0.8167 | 0.8000 | 0.7000 | 0.7100 | 0.2500 | 0.7523 | 0.0167 |
| M1 dual | rain | 0.8583 | 0.8417 | 0.5500 | 0.6100 | 0.3500 | 0.7073 | 0.0167 |
| M1 dual+radius | clear | 0.6417 | 0.4083 | 0.1000 | 0.5800 | 0.0500 | 0.4793 | 0.2333 |
| M1 dual+radius | low-elev | 0.5333 | 0.2583 | 0.0000 | 0.5200 | 0.0000 | 0.3452 | 0.2750 |
| M1 dual+radius | rain | 0.2167 | 0.2167 | 0.0000 | 0.2200 | 0.0000 | 0.2183 | 0.0000 |

M1 dual的旧类保护跨三场景稳定，clear达到零遗忘和90%旧类floor；主要失败变成新类竞争不足，尤其clear new floor仅10%。

## Support锁与门禁

raw dual锁定`T_old=0.9,T_new=0.9,new_offset=0`：

| scenario | old侵入率 | new保真率 | support old floor | support new floor |
|---|---:|---:|---:|---:|
| clear | 0.0167 | 0.6200 | 0.8000 | 0.2000 |
| low-elev | 0.0333 | 0.6200 | 0.7000 | 0.2000 |
| rain | 0.0167 | 0.5400 | 0.8000 | 0.2000 |

radius锁定诊断`T_old=1.1,T_new=0.9,new_offset=0.04`，但没有任何网格配置同时满足三场景门禁。其预选诊断的support old侵入率为35%/66.67%/0%，new保真率为52%/30%/24%，因此标记`SUPPORT_GATE_BLOCKED_DIAGNOSTIC`，不得晋升。

## Loss与资源

- 完整`loss_trace.jsonl`共90条、9组，每组严格包含epoch1–10，0个NaN/Inf。
- 三场景的`theta_B`、pure L6q和M1 `theta_C` loss均从首epoch持续下降；没有训练collapse。
- M1 `theta_C`最终support old侵入率均低至1.67%–3.33%，但new-support floor仍只有40%附近，解释了query new floor不足。

| candidate | trainable params | sequential epoch | persistent state | query classifier MAC | classifier latency |
|---|---:|---:|---:|---:|---:|
| pure L6q | 256 | B10+C10=20 | 29,404B | 29,184 | 同batch统计内 |
| M1 dual | 512 | B10+C10=20 | 30,428B | 30,208 | mean 0.00279ms/sample |
| M1 dual+radius | 512 | B10+C10=20 | 30,472B | 30,219 | P95 0.00436ms/sample |

全部候选低于50k参数、20epoch、256KB状态上限，无dense query图。实测峰值CUDA显存64,986,112B。

## Artifact

| artifact | SHA256 |
|---|---|
| predictions_k10_new5.npz | `d1708e39875bd425949a8539cd0343b5149a6e0fe61c2bacdc8f200dfc50b4ba` |
| predictions_k10_new5.receipt.json | `0ce26fe3141b8ae05cafa63d76865e35375ef1261d233df08cdd3c4dbc7bc212` |
| loss_trace.jsonl | `e2f31bc2d3d4b5abd983fbe9d6ebadc8cbf33fe18f9d0b213c8ef29b69eaa1fe` |
| score_k10_new5.json | `8979ded33746b68c266f5c84bb288edf9d5b0f7e062e8cf460a4e05286763dea` |

## 结论

M1证明双状态可以有效降低多新类注册造成的旧类遗忘，但当前统一校准过度偏向old，牺牲了新类识别。若继续，应直接加强new-fidelity约束或设计类条件但仍support-only的跨空间校准；support radius标准化应淘汰，不应继续调参。当前所有结果远未达到项目正式门槛，不能进入125确认矩阵。
