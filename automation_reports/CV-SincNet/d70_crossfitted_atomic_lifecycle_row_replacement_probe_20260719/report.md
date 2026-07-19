# D70交叉拟合原子生命周期行替换探针

## 1.执行前登记

- 实验ID：`d70_crossfitted_atomic_lifecycle_row_replacement_probe_20260719`；operator：Codex；状态：`PREREGISTERED_IMPLEMENTATION_PENDING`。
- 当前联合最强D62：B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。
- D69完整结果为92.78/81.67/74.67/77.39/11.11/30.00，min-N53.33%、混淆27/23/15；全旧行冻结跨坐标系交换已否决。D67–D69正式复盘见D69报告第15节，提交`6c5f924e`。
- 根目录`E:\type10-7`非Git；本报告镜像、代码、测试和追踪进入`E:\type10-7\github_publish\CVS-RFFI-repo`。其他工作树改动与D70无关，只暂存D70拥有路径。

## 2.方法锁

K>=2时使用两个按physical rank预定的互斥support-held fold。每折在train部分分别拟合D62 before-old和D62 final-joint；held全部类上，以final-joint为base，逐个旧行测试before行替换。单行要求本类TP不降、FP不增且至少一项严格改善；全部初选行联合替换后，必须对11类逐类TP不降且FP不增，否则mask全清零。full support只按mask在D62 final head中替换旧行，新行始终为final joint行。K1精确D62 fallback。

没有连续权重、center/scale、符号、温度、offset、class名单、scene/receiver或query角色分支。所有候选旧行使用同一计数公式；最终是一个全注册类affine head。

## 3.目标、停止条件与完整报告

- before、空mask fallback必须精确D62；两折partition exact-once，gate联合TP/FP原子安全，旧/新类评价同等。
- 相对D62必须无A/N/H/J/min-A/min-N/场景floor交换，并至少改善A/F/J/floor之一；否则`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- INT8相对FP32的argmax变化和margin sign flip为0；资源、query独立性和状态上限通过。
- 真实105行后完整报告7候选、3场景、11类、15fold、mask/TP/FP、训练、量化、资源、artifact及D62/D65/D66/D67/D68/D69对照。失败不跑第二seed/125。

## 4.数据与协议

固定receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8；复用D18`VALIDATED_ONCE/p2_min_v1`enrollment-only capsule，不重验数据。query只测试，no clean/source、query truth/role/quota/global assignment、class-ID规则或dense query graph。ground输入锁0：D22仍`formal_phase2_eligible=false`，D66的84-cell接入为负交换。

## 5.实施计划

新增独立D70 partition/gate/lifecycle core、probe和专项测试，不修改D62/D69历史实现。先做合成partition、原子gate、置换、空mask精确fallback、K1、旧行选择、新行恒定、compiled state、禁止访问和资源闭包测试；再跑D42–D70完整链、提交、干净worktree复跑，最后才登记真实105行命令。

## 6.实现与本地验证

- `code/cvsrffi/stage2_d70_atomic_lifecycle.py`：两折rank partition、TP/FP计数、coordinate gate、all-class atomic gate和Stage2-B/Stage2-C配对生命周期。
- `code/scripts/probe_d70_crossfitted_atomic_lifecycle_row_replacement.py`：复用锁定D62与D42 runner，记录60次top-level fit、30对生命周期、120次inner D62和2280条component fit；单独计入inner LDA/Fisher/held-score/gate MAC。
- 两个测试文件共10项，覆盖partition exact-once、原子安全、置换等变、K1精确D62、选择性旧行、新行joint不变、support漂移拒绝、source closure和禁止分支。
- 专项10/10通过；D42–D70完整链345/345通过，用时81.5s，34个测试文件，包含D42 integration20项。
- 主工作树source SHA：core`f2e67c142ba8fbe797a019e724435a86b67db8446efb9ba49c96abb593b47459`；probe`ff74748be440648ade9c45c60d12c53ea71e149d74180b30a4c1570a257072c2`。

当前只有代码/合成验证，不能声明性能。下一步提交精确文件，建立干净worktree复跑345项；干净链通过后才登记真实105行命令。
