# Phase3 源端星地域读出与冻结开发迁移计划

本候选只改变源端拒识读出训练条件与 known-only 阈值校准，不更新 E51 身份编码器、原六类默认头、原 clean-L 几何或 DURA。主候选事前固定为 **augmented_readout::mlp16_oe**；calibration_only 和其余六方法均作诊断，不根据开发成绩换选。source 科学指标失败也继续完成固定 C0/P1 开发矩阵，不以源 unknown 分数取消或改排。

run 为 `20261029-phase3-source-practical-readout-s2026092701-r01`，单模型种子 `2026092701`。科学配置为 [pamf_phase3_source_practical_readout_20261029.json](E:/type10-7/configs/pamf_phase3_source_practical_readout_20261029.json)。实际执行 commit 在发布前以 `.codex_tmp/pamf_phase3_practical_prepare.py --commit <actual40hex>` 写入 runtime、export／dev 配置及登记 spec；未填写 commit 的本地草稿不可启动。本文是预登记，不预告实际运行或性能通过。

## 数据与两 arm

复用原 source 五 RX（1-19、18-2、19-2、2-19、3-19，index 1/3/4/6/8）与三日（2021-03-08、03-15、03-23），不改变原 L/U/V 契约，不读取原 U_s 或目标 U。source OE 仍为既有原四台训练辅助设备；novel3 仍为原 source 开发验证设备。数据及模型来源按已有 VALIDATED_ONCE／checkpoint 身份复用，不重新扫描 Full、ZIP 或逐成员哈希。

| source 角色 | 原物理记录 | 四视图计算曝光 | 可用监督和用途 |
| --- | --- | --- | --- |
| L_s | 6300 | 25200 | 旧六类可见；二元读出仅使用 known 成员监督 |
| 原四类 OE train | 4200 | 16800 | 仅 source unknown 成员监督，无 TX/y 字段 |
| known V | 27000 | 108000 | 六类标签仅用于风险校准与评分，不拟合读出、不早停 |
| novel validation | 3150 | 12600 | 两 arm 阈值全部冻结后才消费紧凑表示作预测；TX truth 只给独立 scorer |
| 合计 | 40650 | 162600 | 同一物理记录的衍生条件，不是更多独立样本 |

四视图按 `clean`、`practical_low_suburban`、`practical_mid_urban`、`practical_low_urban` 固定顺序。clean 保留已有 source 一次 unit RMS 输入；三个 practical 使用当前发布的 canonical ResidualChannel 后接冻结 DURA，不再次 RMS／EQ。固定 receiver seed 为 `2026102912`、source channel seed 为 `2026102931`，按 scene／opaque ID 决定增强；同一 RX／日期的 receiver session 复用。source 增强与旧开发固定星地的物理实现不声称逐条相同。

每 role/view 只导出 compact NPZ：`ids`、FP32 `[N,100] descriptor`、FP64 `[N,5] static_scores`、FP32 `[N,6] logits`、`baseline_closed`、`rx_i`、`day_i`，只有 L/V 可含 `y`。V／novel 的 baseline 是原 F 对同一输入的闭集预测；L/OE 用 `-2` sentinel，避免无用途的原 F 前向。没有 raw IQ 或完整逐样本 branch bank 输出，也不将 source 行数组部署到开发 predictor。

| arm | 读出与归一化 | 新拟合预算 | 阈值 |
| --- | --- | --- | --- |
| calibration_only | 原 linear／MLP 参数及 mean/std 精确复制 | 0 | 同四视图 known V 新校准 |
| augmented_readout | L4views＋OE4views 上同架构 linear101／MLP1633；mean/std 仅来自这两个训练角色 | 两模型各 200 次 CPU AdamW，全批量原 balanced binary loss | 同四视图 known V 新校准 |

AdamW 固定 lr 0.001、weight decay 0.0001、betas 0.9/0.999、eps 1e-8。源几何始终为既有 E51 原 clean L6300 的聚合量，不能在 augmented views 重拟合。F 为 uniform_oe E51（原 E50 后新增 222 步、累计 11322 步），原默认头保留；G 为 DURA v1.0 R69 E20/4940、266951 参数，`reuse=true, graph=true, native=false, FP32/TF32off`。本轮 source membership 是额外监督，不能将目标端零 TX 标签描述成全流程零标签。

## 风险校准与冻结顺序

每 arm／方法只有一个 global scalar τ，接受规则 `score >= τ`。使用 known V 四视图，按旧六类 macro 同时保护 overall、每 RX、每 view、每 RX×view 共 30 组：FRR≤5%，原 F 同输入闭集准确率减当前开放准确率≤2 个百分点。ties 整组进入，不拆分；用 Fraction 精确预算，分别求每组最高安全阈值后取最小值，避免构造 `Nlevels×30×6` 大张量。额外 encoder 闭集损伤 0.5 个百分点只作诊断，不加入硬目标。

若零拒绝时原 F 损伤已超 2 个百分点，阈值不能修复编码器损伤：取全局最小 score 接收全部，明确 `no_rejection_feasible=false`／保护失败，不伪报 PASS。校准不读取 novel score 或 TX truth，不按 novel 或开发 V 调阈值，也不以 source 科学失败提前终止固定开发诊断。

执行顺序固定为 source export（实际 checkpoint 的无 query 合成 smoke 先行）→两 arm fit 冻结→两 arm known-V calibration 总 marker 冻结→source 四视图全部预测→独立 source score→C0 全 states 预测／独立 score→P1 全 states 预测／独立 score。source 与开发预测期间都核对源状态／读出／阈值不变。source score 对各 view 独立的 30150 条验证记录报告七方法、RX、TX、known 宏保护及未知拒识；跨 view pooled 结果仅作描述，去除将重复物理记录当独立观测的置信区间。

## 跨接收机开发矩阵与判定

C0 与 P1 复用已完成 RX24-1（index 26）、2021-03-15 的同序 864 条物理 ID：known576、unknown288。真实构成只用于已有 builder／独立 scorer 解释，不输入 predictor。source 五 RX 与开发 RX 不交，但开发日期在其他 source RX 已用，不能声称严格跨日期外推。三台开发 unknown 曾在 source RX 反复评价，不是全新 TX 盲测。正式 RX24-18 的 U/Q 不访问。

| 开发条件 | 输入 | states | 新计算与控制 |
| --- | --- | --- | --- |
| C0 | 独立 clean public，无 practical／DURA | original、calibration_only、augmented_readout | original 精确复用旧 clean-transfer D0；当前两 arm 共一次冻结 F 前向 |
| P1 | 旧固定 star received 经相同冻结 DURA | 同上 | original 精确复用旧 transfer 的 P1 D0；当前两 arm 共一次 G/F 前向 |

两 current arm 共用本次同一 descriptor／static5／闭集预测，原 F 同输入 baseline 复用 D0。历史 original 与本次前向若因执行数值存在闭集差异，记录差异计数和比较边界，不为凑齐计数重跑；两 current arm 仍严格同 forward。C0／P1 输入隔离，P1 不读取逐条 clean 参考；两条件仍只有 864 条独立原 Q 记录。本轮 target U=0，没有 target 适应或 R，不能把历史 dataset 的 planned seen/later 标签解释为本 run 已无标签适应的设备。

采用当前 targets 配置：clean 未知 macro 拒识≥95%，星地≥90%，逐主景报告；known 接收率尽量提高，同时 FRR≤5%、原 F 同输入 closed−open 损伤≤2 个百分点。主候选仅 augmented_readout 的 MLP，所有控制／七方法／TX／场景都报告，不据开发结果换 arm、换方法、调阈值或选择性重跑。匿名注册／未知聚类不在本轮，任何 source 单项通过也不代表跨 RX 通过或总 Phase3 完成。

## 执行与预期产物

N607 单 owner，release 为 `releases/<run>-release`，不可覆盖 run 为 `runs/<run>`。Python 环境及既有 source/G/public 路径从旧 transfer runtime 继承；CWD 是新 release。GPU 从 0 至 7 按实际空闲容量选择，最少剩余 4096 MB，每卡最多新增两个本任务 worker，本 driver 同时只运行一个 GPU job；CPU fit／score 设 2 线程，不停止或修改其他 owner。实际卡号、PID、CWD、命令、日志增长及退出从 post-state 报告，启动返回值不算独立运行证据。

单一启动命令为：

```text
<N607 Python> -B -u -m experiments.pamf_phase3_open_world.source_practical_launch --runtime <new-release>/runtime.json
```

`source_practical_launch.commands(runtime)` 是实际 stage 命令来源；runtime 精确使用 `source_practical_launch.FIELDS`，export／dev predictor 各自只接其固定公开配置，truth 只位于 launcher 的独立 scorer 字段。准备脚本只生成本地 runtime 和 `.codex_tmp/pamf_phase3_practical_filled.json`，不登记、不启动或连接远端。

| 输出路径（相对 run） | 完成标记 |
| --- | --- |
| source/export/<role>/<view>/data.npz、complete.json | SOURCE_PRACTICAL_DESCRIPTORS_EXPORTED |
| source/fit/<arm>/{geometry.npz,readouts.npz,complete.json}；source/fit/{training.json,complete.json} | SOURCE_PRACTICAL_READOUTS_FROZEN |
| source/calibration/<arm>/{thresholds.npy,complete.json}、总 complete.json | SOURCE_PRACTICAL_THRESHOLDS_FROZEN |
| source/predict/<arm>/<view>/predictions.npz、总 complete.json | ALL_SOURCE_PRACTICAL_PREDICTIONS_COMPLETE |
| source/score/{scores,complete}.json | SCORED_SOURCE_PRACTICAL_COMPLETE |
| rows/C0、rows/P1 的 predict/{predictions.npz,complete.json} | SOURCE_PRACTICAL_DEVELOPMENT_PREDICTIONS_COMPLETE |
| rows/C0、rows/P1 的 score/{scores,complete}.json | SCORED_SOURCE_PRACTICAL_DEVELOPMENT_COMPLETE |
| logs/{export,fit,calibrate,predict_source,score_source,C0_predict,C0_score,P1_predict,P1_score}.log | 各 stage 完整 stdout／stderr |
| launch.json、pipeline_state.json、observed_state.json、observations.jsonl | 独立 post-state；driver 最终 ANALYZED_DEVELOPMENT_ONLY |

技术停止条件仅为 source／role／数据句柄错误、checkpoint 不匹配、非法 truth／fit／参考访问、非有限输出、不可覆盖路径冲突或不能生成合法完整预测。保留 partial，不按性能重跑、改 seed 或改配方；source／开发科学指标失败如实评分报告。完成既定必要负测、真实 checkpoint 无 query smoke、一次独立 P0/P1、单 commit、一次发布归档比较／远端 compile、运行 post-state 及独立 truth-last，不增加其他审批、hash 链或报告门槛。
