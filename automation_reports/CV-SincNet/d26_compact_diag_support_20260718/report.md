# D26 compact-diag高维拼接support-only实验

## 启动前记录

- experiment ID：`d26_compact_diag_20260718/support_screen_v1`。
- 日期：2026-07-18；operator：Codex；状态：`LOCAL_VERIFIED_PENDING_N607`。
- 目标：压缩D25 v4中性能最强但60 optimizer steps超限的B3诊断结构，在单IQ 288D拼接下用≤30步完成Stage2-B适配和Stage2-C新类注册，并显式保护旧类floor。
- 假设：B3相对C3的主要收益来自逐类可学习cosine head；把mini-batch 40+20步压成全批次15+(0/10/15)步，并增加support-only new-group bias，可保留B3大部分old/new收益，同时把峰值活动参数降到2,016、总步数最多30。
- 比较目标：Z0、B3、C0、D26-A `15+0`、D26-B `15+10`、D26-C `15+15`。
- 矩阵：6候选×3个LEO_weak场景×5个held-rank fold=90行；receiver `20-1`、开发seed `713101`、K=10，每fold每类fit8/held2，旧6类+seen-new5类。

## 本轮不再做数据准备

- 直接复用D25/C3已验证的sealed enrollment-only LEO_weak support，不新增、不重新叠加、不派生任何星地信道样本。
- 每个物理IQ仍只有一个LEO_weak观测；`z160/FFT96/RF32`只是同一接收IQ的一条288D拼接行，`support_view_count=1`、`support_row_multiplicity=1`。
- query/test不打开，不参与适配、bias选择、回滚或ranking。

## 方法与选择锁

- Stage2-B：shared 288维对角+6个逐类288维weight，全批次15步；不更新backbone。
- Stage2-C：只更新5个new suffix weight，分别0/10/15步；旧weight、shared diagonal和old raw score列冻结。
- new-group bias只从预锁定`[-2,-1,-0.5,0,0.5]`由新类support LOO和旧support安全门选择；它作用于注册类身份，不读取query角色。K=1强制0且不伪造LOO。
- fold与full-K10都比较注册前后旧support逐类准确率和floor；任何退化不得晋级。
- 每场景pooled old/new floor相对C0均至少提升10pp、任一类下降不超过10pp、H不低于C0、forgetting不高于C0；B3仅为性能参考。

## 本地版本与验证

- Git仓库：`E:\type10-7\github_publish\CVS-RFFI-repo`；根目录不是Git仓库，本报告在根目录与Git承载面保持镜像。
- D26核心提交：`0a9fbb20`；runner提交：`e4681cee`。
- runner SHA256：`4b664deff293571e44a86c3157f918146adda42e35167adb4d9836f2cbffcb52`。
- D26核心SHA256：`c03d2990a88b1526d40728fa616e4e4e6a43bc42c3e67e24388687aee35d6899`。
- launcher SHA256：`d49e7626a90b0c2b068f83651d5760033365c8b2cd60401769822f3b6434a2e3`。
- 55项D26+D25/C3相邻回归PASS；`py_compile`、`bash -n`、`git diff --check`PASS。
- 独立review未发现协议或算法高严重度阻断；已修复D26实现Git归因和FFT96/RF32 operator闭包遗漏。

## N607计划

- 已有2026-07-18 01:16 CST直连preflight PASS；正式启动前重新读取live process/GPU inventory。
- 远端根目录：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 计划GPU：优先空闲GPU0，但不超过每GPU两个训练任务。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d26_compact_diag_20260718/support_screen_v1.log`。
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d26_compact_diag_20260718/output/support_screen_v1`。
- 精确命令：`cd /home/szu2070436088/2510044040/CV-SincNet && D26_GPU=0 bash code/scripts/launch_d26_compact_diag_support_20260718.sh`。
- 只同步runner、D26核心和launcher；不会覆盖远端现有`stage2_diag_cosine_exploration.py`。launcher与runtime candidate lock会校验并记录远端实际operator SHA=`14ec919395f9bf9f13214c677b1a3d640764214668d1d00e9109f5b149ec41ca`。

## 预期产物与判定

- 六件固定artifact：`training_log.jsonl`、`support_audit.json`、`selection.json`、`resource_audit.json`、`geometry_audit.json`、`RECEIPT.json`。
- 完成后报告候选联合行、逐场景、逐类old/new floor、bias选择分布、注册前后旧support、完整loss、参数/step/状态/MAC/时延/显存和相对qKNN Pareto。
- 本轮仍是开发support-only筛选，不是正式query性能；只有正向路线才进入joint bundle method lock和正式5receiver×确认seed×3scene矩阵。
