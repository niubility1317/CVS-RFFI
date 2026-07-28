# Findings & Decisions

## 2026-07-28初始证据

- 活动目标已存在，目标文本与用户本次请求一致。
- `E:\type10-7`根目录不是约定Git承载面；正式计划、追踪和实现放在`E:\type10-7\github_publish\CVS-RFFI-repo`。
- 设计报告已由提交`790da982`镜像到`paper/ieee_transactions_draft_20260727/experiments/CVS_FULL_ABLATION_DESIGN_PHASE1_PHASE2_20260728.md`。
- 发布仓库从`codex/cvs-rffi-release-20260626`的提交`98d0898`创建本目标分支`codex/full-ablation-20260728`；工作区仍存在大量历史未跟踪测试目录和`local_artifacts`，不得清理或误提交。
- 当前Stage2候选worktree`ground_proto_da_rd_wt`含一项已跟踪测试修改和多项未跟踪实现/测试文件，归属不明；在审计前不得覆盖或直接用作本次发布基线。
- 本地对话索引已刷新为1188条。检索只定位到D92历史审计和本次新目标，没有发现本设计已完成运行的历史证据。
- D92历史125矩阵只能作为实现/回归线索，不能替代本设计要求的fresh screening或confirmation。

## 设计规模

- Phase1第一层：6个arm×5个训练seed=30次完整训练。
- Phase2 screening：每arm 75个注册row、225个场景单元。
- Phase2 fresh confirmation：每arm 900个注册row、2700个场景单元。
- “全量”是分层漏斗，不是一次性全笛卡尔积；T2内部arm只有在对应T1整体作用成立后才运行。

## 当前最高风险

1. 报告第13节列出的开关/arm factory/连续状态/resource profiler可能仍有实现缺口。
2. `P1-FULL`必须在当前`0.07/0.63/0.30`划分重训，历史`ADV3B02_CORE90_SOFT_E200`不可直接替代。
3. `P2-FULL`名称对应RTB-IDR/D92闭环，但历史D92并不等于本设计的fresh完整确认。
4. 8×2调度器必须把服务器已有任务计入每卡2进程上限，并产出不可覆盖行级artifact。

## 主线程实现审计

### Phase1

- `code/SSDG/train_ssdg.py`已经暴露domain/GRL/orth/center consistency、GroupCE、FishR、伪标签门、EMA、satellite CE/consistency、source episode、prototype及多种几何控制参数。
- `code/model.py`已经支持`no_time/no_dac/no_pa/no_freq/no_stats`组合；`code/model_dual_cvsincnet.py`支持identity/domain分别使用`branch_ablation`、domain RCN enhancer开关和`cvcnn/sinc_cvcnn`架构族。
- 这些底层开关尚未证明存在统一`ablation_id`factory、one-factor config diff、参数量匹配A0/Conv对照、matched-coverage B1构造器或报告要求的完整指标/resource artifact。
- `train_ssdg.py`的CLI默认划分仍是`0.08/0.72/0.20`，与本设计正式`0.07/0.63/0.30`不一致；正式arm必须由锁定配置显式覆盖并由validator拒绝漂移。

### Phase2

- D92正式retry2代码身份是提交`87012f4138c1cd308468ef74e238131af949c651`，不是`d92_125wt`当前Role-Oracle HEAD。
- 发布仓库当前缺少正式D92的5个关键文件：核心协方差、query evaluator、probe、125 runner和summarizer；不能从当前Oracle HEAD复制。
- 正式D92复用了D81稳健中心、D43/D45/D62双几何/融合/Fisher链，并在注册后且`K>2`时启用旧/新任务等权协方差；注册前与K1/K2精确回退。
- 现有D92 runner锁定为125 job、8 shard，历史运行每GPU只新增1个D92分片；它不是本目标要求的75-row screening、900-row confirmation或16并发worker调度器。
- 历史retry2完成125/125且0失败，但状态是`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；只能用于实现回归和缺陷史，不能替代fresh实验。

## 三方只读审计裁决

- Phase1：六个第一层arm中0项已验证、5项部分实现、1项缺失；`P1-A0`参数量匹配单embedding是真实P0阻塞。旧launcher的`0.10/0.70/0.20`不能替代当前`0.07/0.63/0.30`重训。
- Phase2：历史prediction/truth分离已实现；P2-FULL、A–F模块、7基线、K2、75/900矩阵、连续注册和16槽调度均只部分或未实现。旧D92 retry2只作负向诊断回归。
- 监督审计：当前裁决`NO-GO / 禁止发布N607`。必须先完成本地T0、真实checkpoint无query smoke、fresh registry、Git提交和独立`P0=0,P1=0`审查。
- `P2-F3`与`P2-FULL`是同一完整量化状态；矩阵保留两个逻辑ID用于论文表语义，但物理执行去重并共享同一不可变artifact引用，避免虚假重复实验。

## 已冻结的第一批身份层

- 新增`code/cvsrffi/full_ablation_spec.py`，锁定Phase1 30-row、Phase2 75/900-row、8GPU×2槽、fresh seed、防复用和必需artifact字段。
- 新增`configs/full_ablation_20260728/seed_registry.json`。候选seed已对1188条项目对话索引、Git跟踪面和自动化报告控制文件做精确值搜索，未见历史使用；旧`713101–713106`显式拒绝。
- 新增非启动型计划构建器`code/scripts/build_full_ablation_plan.py`。验证得到Phase1 30个物理row、Phase2 T1 screening 1425个逻辑row/1350个唯一物理row、单arm confirmation 900个row；所有计划保持`formal_launch_authority=false`。
- 首组规范测试在`ssr-gpu`下7项通过；这只证明身份、计数、seed和调度边界，不等于真实executor就绪。
