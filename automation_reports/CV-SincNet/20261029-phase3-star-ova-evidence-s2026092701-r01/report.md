# PAMF 第三阶段：类条件开放边界与难负类学习

当前登记状态：ANALYZED_SOURCE_ONLY；两stage COMPLETE，实测NO_PROMOTION。以下预登记保留，结果见末节。

## 预登记

Run `20261029-phase3-star-ova-evidence-s2026092701-r01`，code `0655a5cb0975dc047c712a76630d3e0354da0e55`；唯一主方法`ova_max`，固定四决策membership_routed/membership_max/ova_routed/ova_max来自两个共享MLP，不以控制替换主方法。方法与门槛见`docs/PAMF_PHASE3_STAR_OVA_PLAN_20261029.md`和`configs/pamf_phase3_star_ova_20261029.json`。

F0是原三视图Main split E50/11100/default6head/232993参数，所有身份仍原P1 argmax；P0/P1共享每记录一份G10 received，P1仅复用冻结DURA v1.0 R69 cache。本轮source L6300/OE6134拟合两个同initial106×32×1共享reader（各3457参数/1000 CPUupdates，总2000）；L真实六标签供OVA，OE仅member/domain不读TX。共同normalizer .6/.4 roleprior、15domain等权、六classslot等权；F/G/head/geometry后续不变，F/head/G0updates。V27000仅四组六predclassτ，42组CK90/F0loss2pp，全部τ冻结后source novel3150才打开。所有四预测完成后独立sourceprivate truth评分，不从novel选模或重校准。

source五RX1/3/4/6/8、days2021-03-08/15/23，目标RX不在config；same sourceRX和历史曝光三novel，仅source可证伪研究，不能称目标跨RX成功。预测class_scores[N2,6]可重建routed/max与24τ，错分known是否接收与身份正确分开。主报trueTXmacro AK/RU/H，并列CK/originalF0closed/loss、AUROC/OSCR、TX/RX/scene/CI和配对净数。所有known42保护且主ova_max sourceunknown90 overall+5RX+6scene及必要覆盖全过才允许未来另run目标falsification，否则NO_PROMOTION。AN/HOW为N/A，完整Phase3仍ACTIVE。

### 精确执行

- Host N607，Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-ova-evidence-s2026092701-r01-release`，唯一owner `codex-phase3-star-ova-20261029`。
- 命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_ova_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-ova-evidence-s2026092701-r01-release/runtime.json`。
- Runtime source config `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-ova-evidence-s2026092701-r01/configs/source.json`；输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-ova-evidence-s2026092701-r01/prediction`、评分 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-ova-evidence-s2026092701-r01/score`、日志 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-ova-evidence-s2026092701-r01/logs`；local `E:/type10-7/local_artifacts/pamf_phase3_star_ova_evidence_20261029_r01`。
- Cache `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache`四role已有VALIDATED_ONCE/STAR_REPRESENTATION_CACHE_COMPLETE；source contract `/home/szu2070436088/2510044040/CV-SincNet/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/source_contract.json` countsL6300/Us56700/V27000、split392005，匹配F0 ancestor。不新读Us/raw/真实信道重生。
- F0 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/final_ssdg.pth`，base release `/home/szu2070436088/2510044040/CV-SincNet/releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release`，frozen source `{'F': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_supcon50_clean_20261016_r02', 'AF': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RDrop_AF50_20261015_r02', 'BASE': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02', 'FOLD': '/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01', 'G': '/home/szu2070436088/2510044040/CV-SincNet/releases/pamf_stage2_inference_20261021_r01'}`；cachedG身份由role manifest核对，不新训/新前向G。
- Source private novel truth仅CPU scorer：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-representation-s2026092701-r01-release/score_truth/source_novel.npz`；没有目标U/Q/trueK/配额句柄。
- one ownGPUworker，actualfree≥4096MB、每卡≤2added；F0 GPU/两个reader CPU/独立score CPU，其他owner不修改/停止/重启。

预期artifacts：launch/pipeline/logs/config；prediction complete/smoke/readouts NPZ+JSON、两geometry、24conditional+4globalτ、calibration complete；known/novel各class_scores与margin4；score results/complete。F0实际85170行（含2 synthetic G10 smoke），真实G新0，source42584物理记录双path不算独立更多样本。

验证：23 tests/55 subtests、四AST与实际runtime schema通过；一次独立P0/P1审查，现有pipeline无重复审核/报告门槛。一次N607preflight输入匹配、GPU足够，真实F0 synthetic-G10 first-step noquerysmoke PASS立即继续；archive一次SHA比较/remotecompile；启动后独立PID/CWD/argv/GPU/log增长，完整预测后truth-last。

技术停止只限祖先/角色越界、输入输出碰撞、非有限或无法完成合法预测，保留partial不原地重启；低分或缺覆盖是科学结果，不调参重跑、不访问target。当前只有预登记，无性能结果或完成声明。

### 发布前核实

唯一独立P0/P1审查已PASS，未发现下一真实run的直接阻断；独立定向7tests/22subtests与主测试重叠不加计unique。核对两模型预算、原P1身份、novel开启前阈值冻结、四分数重建和独立truth顺序，控制不替代主方法。一次N607 preflight通过，sourcecontract392005和缓存四role身份/数量匹配，actual显存充足、target句柄0。代码提交0655a5cb0975dc047c712a76630d3e0354da0e55；启动与性能证据待后续真实artifact读回。

## 2026-10-29 G10 源端共享OVA证据实测结果

`20261029-phase3-star-ova-evidence-s2026092701-r01` source_predict/score全部COMPLETE；唯一主ova_max **NO_PROMOTION**，全目标 **ACTIVE**。

| scalar | AK% | RU% | H% | CK% | F0损pp |
| --- | --- | --- | --- | --- | --- |
| membership_routed | 90.4519 | 38.6349 | 54.1434 | 98.4111 | 0.5296 |
| membership_max | 90.4000 | 40.8254 | 56.2485 | 98.3556 | 0.5815 |
| ova_routed | 89.9259 | 58.9206 | 71.1940 | 97.3556 | 1.0556 |
| ova_max | 90.0185 | 55.5873 | 68.7320 | 97.6889 | 0.9630 |

四scalar均known42通过，主实际unknown必需12/12全部失败。OVA−membership同routed正确−142/未知拒识+639，同max正确−103/未知拒识+465；OVA max−routed正确+25但未知拒识−105、H−2.4621pp。四scalar各自known-only校准，不把差全部归因纯max算子，不换控制当主。
两sharedMLP3457各1000CPU步、同初始权重，总2000；normalizer仅L/OE，成员监督+known真实L类用于OVA目标；V27k仅42组/24τ，novel3150最后预测/独立truth-last。四scalar由class_scores[N,2,6]重建验证，identity原P1，不是nativeOVANet/targetOEM/六身份头。
源85168F0角色行＋2synthetic=85170，Gnew0/NNencoder更新0，无新物理数据或target访问。主最小503401参数、实际两读出506858；两geometry/212normalizer/24τ另计。source83.421217s/score3.287446s，pipeline86.790642s非逐包时延。独立sourceGPU4/388MB两次离散样本、log9782→11034→21611B；terminalalljobs0/live0/foreign0。
完整四方法/42组/真TX/2000loss/配对净数/成本和资源证据见[`PAMF_PHASE3_STAR_OVA_RESULTS_20261029.md`](../../../docs/PAMF_PHASE3_STAR_OVA_RESULTS_20261029.md)。同源5RX和历史novel不是跨RX target成功，AN/HOW=N/A，不新增实验。
