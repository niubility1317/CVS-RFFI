# PAMF 第三阶段：全 U 邻域适应与匿名注册

当前登记状态：ANALYZED_DEVELOPMENT；三stage COMPLETE，实测NO_PROMOTION。以下预登记保留，结果见末节。

## 预登记

Run `20261029-phase3-star-neighborhood-s2026092701-r01`；唯一主状态 `DA1_REG1`，四态固定比较。方法和数据详见 `docs/PAMF_PHASE3_STAR_NEIGHBORHOOD_PLAN_20261029.md` 与 `configs/pamf_phase3_star_neighborhood_20261029.json`。只在全 U 上选择无标签聚类划分，未来 Q 不拟合。原 F0/DURA/paired1000/class-risk 六τ冻结，源码身份 `cb58217de28eb2cb5536d2392aef268b79006877`。源42组已知保护由已冻结 parent calibration 核实；这次小开发矩阵不表示旧 source-only 拒识失败候选已晋级 formal。

本轮新源输入为已有 G10 L6300，每条P0/P1共享，90个TX/RX/day均值anchor部署；原 readout 的 L6300/OE6134 成员监督及 V27000 标签阈值祖先全部披露。无新OE/V/novel/原Us输入。目标RX24-1 U384/futureQ864，总1248原物理记录，源/目标RX不交、U/Q物理互斥，原six-component G10 MIX01不重抽。真TX/角色/K/配额只在builder/scorer，目标predictor只公开received/opaque IDs。正式RX24-18=0。

当前指标为v3：class macro A_K/R_U/H-score；C_K≥90%、R_U≥90%、原F0同P1闭集至open损失≤2pp，适应同Q旧AK正收益；未来AN≥80%、注册旧类较sameDA_REG0损失≤2pp、intendedCK≥90%、后续未知R≥90%。总体和六景均检查，缺必需TX覆盖为N/A且不可晋级。四态固定全部执行，科学失败不重选K/阈值/状态、不重跑、不扩种子或formal。Q的匿名映射只用Utruth拟合冻结，一对一，未匹配anon接受算错误，未发现真实到达类仍在AN分母。

### 执行

- Host N607；Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-neighborhood-s2026092701-r01-release`。
- 精确命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_neighborhood_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-neighborhood-s2026092701-r01-release/runtime.json`。
- 唯一 owner `codex-phase3-star-neighborhood-20261029`；one GPU worker at a time，actualfree≥4096MB、每卡≤2新增，foreign工作不改。源和目标GPU步骤独立重查显存，score CPU。
- Source config `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-neighborhood-s2026092701-r01/configs/source.json`；target config `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-neighborhood-s2026092701-r01/configs/target.json`；输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-neighborhood-s2026092701-r01`；本地取回 `E:/type10-7/local_artifacts/pamf_phase3_star_neighborhood_20261029_r01`。
- 源L `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache/L`；父readout `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-paired-evidence-s2026092701-r01/prediction`；父threshold `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-class-risk-s2026092701-r01/prediction`。
- U `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-relative-s2026092701-r01-release/data/star_g10/discovery/public`，capsule `p3dev-star-g10-6e1292b62ab09286ebf1a0df`，split `83c5cc1fe2f2a1512dbc1287`；Q `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-relative-s2026092701-r01-release/data/star_g10/evaluation/public`，capsule `p3dev-star-g10-71a5a0dec23ac6ccc3b479c2`，split `05de65cd4818f3e5e33ced5c`，复用VALIDATED_ONCE，仅检查最小handles，不重验数据。
- 独立评分Utruth `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-relative-s2026092701-r01-release/score_truth/U.npz`；Qtruth `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-relative-s2026092701-r01-release/score_truth/Q.npz`；target/source predictor配置不含truth。阈值 `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-neighborhood-s2026092701-r01-release/configs/pamf_phase3_targets_20261029.json`。

预期：source complete/smoke/90anchors/两geometry/readout/阈值；prediction四态/registry冻结/U诊断；score scores/complete；launch/pipeline/全部日志和独立observation。源第一步actual F0 syntheticG10无querysmoke，目标actual F0/G synthetic无querysmoke，PASS后立即继续。

聚焦验证：24 tests，34 subtests（按实际测试输出记录，重叠不加计）；一次独立P0/P1审查结论以对应消息/事件为准。一次N607preflight、release单archive一次SHA比较/remotecompile，启动后PID/CWD/cmdline/GPU/log增长核验；不是以启动返回值认定成功。

技术停止：输入/祖先/角色不匹配、越界、碰撞、非有限、无法完成合法预测；保留partial不原地重启。科学低分/零簇/不完整场景不重试，不改其他owner进程。当前预登记阶段，无性能完成声明，完整Phase3仍ACTIVE。

### 发布前核实

唯一独立P0/P1审查已PASS（phase3_release_review，源码READY后一次最终审查）；独立定向5tests/9subtests与主集成测试有重叠，不加计为新unique测试。原F0/DURA、source90anchor、全U无标签聚类、registry先冻结再Q、U-only映射和v3门槛未发现直接阻断。N607 sklearn1.7.0的metric参数接口独立读回支持；一次输入preflight通过，U/Q最小capsule/split/VALIDATED_ONCE匹配，未读取truth内容。代码远端OID与本地cb58217de28eb2cb5536d2392aef268b79006877一致，VERIFIED。新实验启动和性能结果仍待后续独立产物。

## 2026-10-29 G10 无标签邻域四态实测结果

`20261029-phase3-star-neighborhood-s2026092701-r01` source/target/score全部COMPLETE；唯一主DA1_REG1 **NO_PROMOTION**，总目标 **ACTIVE**。

| 状态 | AK% | CK% | RU%/cohort | H-score% | AN% | F0损pp |
| --- | --- | --- | --- | --- | --- | --- |
| DA0_REG0 | 65.9722 | 88.3681 | 76.3889/[6, 7, 8] | 70.7995 | N/A | 6.9444 |
| DA1_REG0 | 61.6319 | 86.6319 | 79.1667/[6, 7, 8] | 69.3074 | N/A | 11.2847 |
| DA0_REG1 | 65.7986 | 94.9653 | 14.5833/[8] | 23.8751 | 37.5000 | 7.1181 |
| DA1_REG1 | 61.2847 | 96.3542 | 14.5833/[8] | 23.5603 | 39.5833 | 11.6319 |

全U silhouette选3簇，仅1匿名簇(93条)符合两半负margin/size条件；旧52/到达label6有41，污染55.9140%。U-only Hungarian只映anon0→6；到达7未发现，仍计AN两设备分母。主AN39.5833%=label6未来76/96与label7零准确率的宏均值。
DA1_REG0−DA0_REG0旧正确净−25/576(−4.3403pp)，未知拒识+8/288；实际融合812/864Q但未成为有效适应。注册DA1较自身REG0旧接受+56、正确−2，固定later拒识81→14/96(−67)，不能把REG0全部三nonold与REG1单later RU直接同分母比较。
主oldCK96.3542%中555接受只有353身份正确；intendedCK87.8906/AN39.5833/laterRU14.5833/F0损11.6319pp均不足。high景缺label6，AN/H/intendedCK N/A、不能认证；完整四态六景/真TX/RXday/CI及净数见结果文档。
本轮实际15102 F0/1250 G前向、NN更新0；源新L只90anchors，无新OE/V/novel。实际四快照512876参数，最小主503177另列；source8.9811s/target9.4034s/score3.4601s，pipeline22.004161s非逐包时延。独立driver/target live CUDA5/GPU508MB离散采样、log524→875；source未livecapture，终态owned/live0/foreign0。
全目标ACTIVE，单开发RX/seed/episode且历史设备/物理暴露，正式RX24-18未访问。完整方法与实际verification见[`PAMF_PHASE3_STAR_NEIGHBORHOOD_RESULTS_20261029.md`](../../../docs/PAMF_PHASE3_STAR_NEIGHBORHOOD_RESULTS_20261029.md)，不扩展新训练或改阈值。
