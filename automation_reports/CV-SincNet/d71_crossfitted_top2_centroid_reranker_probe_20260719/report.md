# D71交叉拟合top-2局部中心重排探针

## 1.执行前登记

- 实验ID：`d71_crossfitted_top2_centroid_reranker_probe_20260719`；operator：Codex；状态：`IMPLEMENTED_LOCAL_CHAIN_VALIDATED_PERFORMANCE_PENDING`。
- 比较目标D62：B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。
- D70最终与D62全部汇总/floor持平，但旧→新多2次且额外计算显著；提交`e9549c7e`。旧类行替换路线停止。
- 根目录`E:\type10-7`非Git；本报告镜像、代码、测试和追踪进入`E:\type10-7\github_publish\CVS-RFFI-repo`。只暂存D71拥有路径，不覆盖其他工作树改动。

## 2.方法锁与假设

D71始终保留D62全类joint分数，只允许经过两折support-held pair非劣门和全类TP/FP原子门的最近中心pair，在D62当前top-2内部交换次序。它不能引入第三类，不改D62单类行，不做全pair投票或全局score融合。pair公式统一、标签置换等变；K1与空mask精确D62。

假设：D62的主要损失来自少数局部碰撞，而不是全局类别几何整体错误。低方差pair中心只处理D62已认为最相近的两个候选，可能修复old4/old5/new1/new3，同时避免D64全pair锦标赛对全部决策的系统性改写。

## 3.数据、协议与资源

固定receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8；复用D18`VALIDATED_ONCE/p2_min_v1` enrollment-only capsule，不重验数据。query只测试，no clean/source/query truth/role/quota/count/global assignment。ground输入锁0，因为D22仍`formal_phase2_eligible=false`。

部署目标为D62 int8/FP16 head加稀疏int8 pair方向；每query最多增加一个288D pair dot及top-2排序，状态上限256KiB，dense query graph为0。所有额外fit、MAC、pair状态、INT8/FP32差异和实测时延都必须单列。

## 4.验证、运行与停止门

先完成core的partition、pair方向、类置换、top-2第三类不变、原子门、K1/空mask、INT8/FP32和非法输入测试；再接入锁定D62、运行D42–D71回归链并在干净worktree复验。只有这些通过才登记真实105行命令。

真实完成后必须报告7候选、3场景、11类、15fold、接受pair、held TP/FP、训练20epoch、量化、资源、artifact及D62/D65–D70同row对照。相对D62若A/N/H/J/min-A/min-N或场景floor发生交换，或没有至少一项严格改善，则状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不跑第二seed或125，不扫描pair阈值/权重/温度。

## 5.实现与本地验证

- 新增`code/cvsrffi/stage2_d71_top2_centroid_reranker.py`：exact-once两折、最近中心pair方向、逐pair两类非劣门、全类TP/FP原子门、INT8/FP32稀疏pair状态和top-2交换器。
- 新增`code/scripts/probe_d71_crossfitted_top2_centroid_reranker.py`：接入锁定D62，保留D62 base state；对before/final分别登记pair gate/state，并显式增加适配MAC、query额外计算、pair状态和非单affine口径。
- 两个D71测试文件共12项，覆盖partition、pair registry、第三类不变、空门、K1、INT8/FP32、active gate、非法support、D62 audit包装、state identity、协议与调用闭包；12/12通过。
- D42–D71共36个测试文件、357/357通过，用时82.8s。pytest退出后仅出现Windows临时`pytest-current`清理`PermissionError`，命令exit0且全部测试已完成，判为已知包装清理噪声。
- 当前尚无真实outer性能；下一步提交精确D71文件，建立干净worktree并复跑357项。干净链通过后才登记105行命令。
