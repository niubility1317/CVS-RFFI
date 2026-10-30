# PAMF 二元图独立评分恢复

## 预登记（2026-10-30）

新恢复run `20261030-phase3-star-binary-graph-score-recovery-s2026092701-r01`，实际score code `ca05fae1fbdd9b0174e8fb0804ffc2645c5516c5`。原producer `20261030-phase3-star-binary-graph-s2026092701-r01` code`bb9f5823f53bdbba6dd3e65f9f58dfe70bf86cdb`在原root的四态预测／真实F0-G合成smoke已完成，原pipeline仍FAILED_PRESERVED，不覆盖或重启。此次只新独立CPU评分，预测/图/注册/阈值/数据全部原值，0新IQ/权重读取、0NNoptimizer、0预测重生成、0GPUworker。

原failure在独立CPU公有gate、任何U/Qtruth之前；实测unit/ID/sigma完全同，fmaxabs7.77156e-16、残差diag差1.776e-15、jointmax4.44e-16，所有四态pred/closed/接收符号改变0。直接修复只对有限FP64 f/诊断/joint重建用abs1e-12/rtol0；dtype/shape/ID/unit/sigma/离散和最终决策、保存joint内部关系继续exact，并用已验证保存f做下游重构。1e-8实质偏差仍拒绝，不放宽任何科学阈值。

真实公有完整gate本地PASS，无truth或产物变化；RED→GREEN受影响3项PASS5.28s，新增CPU恢复执行3项PASS4.36s，定向AST／镜像通过。只做该直接P1和其CPU恢复执行的定点复审PASS；不重复原26项整体测试或第二次完整review。

原方法数据、单模型seed、源祖先、G10、四态、主DA1REG1、v3各门槛及N/A覆盖政策完全沿用`docs/PAMF_PHASE3_STAR_BINARY_GRAPH_PLAN_20261030.md`和原experiment.json；本run矩阵是原四态全部评分，无候选筛选。源RX1/3/4/6/8与devRX24-1互斥，原U384/Mar8和futureQ864/Mar15物理互斥，全部postPhase1 IQ原已G10叠加，本次不读IQ/新生成。正式RX24-18仍不读。

流程为新独立CPU进程验证原producer完成与公有输出→Utruth两个匿名映射全冻结→Qtruth固定预测评分。未知真K/标签/配额和映射不回算法，futureQ不调阈值/拟合/重匹配。scores保留原producer run/code，recoverypipeline分别记录本run/score code，不把修复版本改标为新训练模型。原producer已成功真实checkpoint smoke为其证据，CPU scorer不消费checkpoint，不额外装载F/G。

- Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-binary-graph-score-recovery-s2026092701-r01-release`；owner`codex-phase3-binary-graph-score-recovery-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_binary_graph_score_recovery start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-binary-graph-score-recovery-s2026092701-r01-release/runtime.json`。
- 新不可覆盖root`/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-binary-graph-score-recovery-s2026092701-r01`；只读原`/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-binary-graph-s2026092701-r01/predict`。预期launch/pipeline/configs/recovery.json/logs/score.log、score/scores.json/complete.json、observations；真实数据/逐行预测/日志保持private。
- CPU CUDA_VISIBLE_DEVICES为空，新增GPU0，foreign干预0。单preflight核原predictcomplete/smokePASS/原FAILED保留及新root不碰撞，私有truth只检查在位不读值。一次code commit/push OID、一次archive SHA/remotecompile，post独立PID/CWD/argv/log/complete证据。
- 同一错误再次出现，保留本run失败，不再盲目恢复。科学结果低分不触发重选/重跑。完整goalACTIVE/finalfalse，不自动扩seed/formal，不修改原run技术失败状态。
