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

## 实际结果：独立评分已补齐，完整目标未达（2026-10-30）

原producer code`bb9f5823f53bdbba6dd3e65f9f58dfe70bf86cdb`四态预测、U图/registry、真实F0/G合成smoke完整；其pipeline仍FAILED_PRESERVED。首个独立CPU在任何truth前因FP64 exactgate误拒，unit/ID/sigma逐值同，fmaxabs7.77e-16/residualdelta1.776e-15/jointmax4.44e-16、全部预测/closed/接受符号改变0。一次定向修复abs1e-12/rtol0只用于浮点重建，原数值/模型/τ/数据/四态pred全部保持。

新独立CPU恢复`20261030-phase3-star-binary-graph-score-recovery-s2026092701-r01` code`ca05fae1fbdd9b0174e8fb0804ffc2645c5516c5`已VERIFIED/ANALYZED_DEVELOPMENT；公有gate→U两个匿名映射全冻→Qtruth，唯一主DA1_REG1=NO_PROMOTION。scores保留原producer身份，recovery另记执行版本；不把原failedpipeline改称成功。source新IQ/逐行feature0、目标标签/K/配额输入0、Qfit0、formal0、新NN参数/更新0。

| 状态 | 宏A_K | 旧类C_K | 当前R_U | H-score | 未来A_N | H_OW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DA0_REG0 | 65.972222% | 88.368056% | 76.388889% | 70.799458% | N/A | N/A |
| DA1_REG0 | 68.229167% | 90.798611% | 57.291667% | 62.283887% | N/A | N/A |
| DA0_REG1 | 65.972222% | 91.493056% | 65.625000% | 65.798153% | 4.687500% | 12.308736% |
| DA1_REG1 | 68.229167% | 93.750000% | 15.625000% | 25.427019% | 19.270833% | 22.979986% |

REG0 R_U为所有未来非旧6/7/8，REG1主R_U为U未出现later8，不能直接以两态H/RU差宣称注册影响。主intended旧六+全部两到达类接收82.031250%<90，不能以old-only C_K93.75%冒充全注册接收达标。原F0同P1闭集420/576=72.916667%，两DA闭集逐值保持；主开放393/576=68.229167%，纯误拒损27/576=4.6875pp>2，无身份漂移。

DA1REG0相对DA0REG0：旧正确380→393净+13（21得8失），接受509→523净+14（25得11失），未知拒对220→165净−55（2得57失）。旧C_K总体90.798611%通过单项，但R_U57.291667%和原F0保护均失败。later拒对64/96→16/96，REG后15/96；同DA REG旧正确保留393→393损0，但旧接收523→540多17不代表身份改善。未来新类主仅37/192=19.270833%，远低80；U簇数DA0=1/DA1=2由算法自主而非真K输入。

二元U种子正136=128old+8unknown，成员精度94.117647%；负27=17old+10unknown，成员精度仅37.037037%，其中旧class4有12。这些是全部预测/独立评分后truth-side诊断，未回算法。低MSP并不能当可靠unknown标签，冻结identity和精确求解本身不足以保护未知。U图1197边，energy128.742233776→37.923038253、残差1.24345e-14、f在[-1,1]，一次solve数值成功不等于科学成功。

数据沿原source L6300/Us56700/V27000/OE6134监督祖先、五sourceRX与目标RX互斥；本轮当前U384Mar8/Q864Mar15均RX24-1、物理互斥、原G10 MIX01与同一固定received。所有postPhase1新IQ均星地，P0received/P1冻结DURA，旧source原权重祖先保留，不读source逐行bank。已暴露devRX/设备/Q，不是新盲测/未见日期或正式RX。high suburban缺必需unknown6，N/A/覆盖false，不重抽样凑指标。

实际F0 full2500=2496real+4smoke、G1250=1248real+2smoke、额外原head smoke2；四个3233读出每次都存/算，real4992/synthetic8MLP行，神经驻留512876无新adapter。合法U memory数值251904B+IDs59904B=311808B，npz312784B/json488B；新CPU恢复仅公有图重构验证和独立评分，无部署新拟合/原预测变动。源训练/恢复器/聚合缓存祖先成本保留。无B1/吞吐/峰值显存或提速证据。

完整4stdout已读取：原predict931B/driver17B、旧score2804B包含已记录技术traceback，恢复score837B无异常。原predictjob9.218162537s/internal5.381406529s、原失败pipeline12.263982534s，CPU恢复score0.501980066s，不能拼成连续端到端或B1。原driver3286708独立live CWD/argv核实，predict3286801/GPU4与旧score3286873只jobs/log支持，未抓liveGPU；恢复CPU3410600独立live正确CWD/argv/CUDAempty，终态live/GPU空、foreign干预0。

初始26tests/24subPASS7.51s和一次freshP0P1；直接数值修复3受影响PASS5.28s/真实公有gatePASS、CPU恢复执行3PASS4.36s与唯一问题定点复审PASS，不累计成同一次32test全suite。原397file/data0 archive d0244e8fdab1199d1fe4da1193108e731e861aaf3e099d884449c2798d72cc1c；恢复399file/data0 archive8d1417794f1a48914a2faf147b52435c7a6682b329287713dca6d24e1963798e，各一次SHA/compile/runtime核实，无重复训练/预测或新数据。

详细四态/24scene/主逐TX与CI、U组件纯度/匿名映射、数据权限、全部成本和失败恢复证据见`docs/PAMF_PHASE3_STAR_BINARY_GRAPH_RESULTS_20261030.md`。本轮不晋级、调τ/seed/K/复制控制，不扩大正式/多seed；全Phase3 goalACTIVE/finalfalse。下一独立科学问题是可靠unknown种子及跨日类别覆盖，不能把低置信度旧样本直接定义未知。
