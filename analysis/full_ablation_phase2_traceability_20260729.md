# Phase2全量消融实现与发布追踪

设计来源：`CVS-RFFI_全部消融实验设计_Phase1_Phase2_20260728.md`

当前目标：在Phase1 T1与标签率任务运行期间完成Phase2 T1的真实执行链准备；GPU槽位释放且P1-FULL deployment bundle完整后，直接发布筛选矩阵。设计中的“75-row screening”是每个arm的75个注册row，即`5receiver×5slice×3development seed`；每row包含3个场景。复用`VALIDATED_ONCE`的Phase2数据切片，不重复数据hash、allowlist、provenance或全量数据审计；不同发布批次可以绑定不同既有合法缓存。

状态定义：`pending`、`implemented`、`verified`、`deferred`、`rejected`、`blocked`。

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|P2-TR-01|7.1|Stage2-A零标签冻结部署独立成表|factory；executor；state plan|verified|Stage2-A K0无support预测单测；25-row独立state plan单测|真实checkpoint no-query smoke待P1-FULL闭合|
|P2-TR-02|7.1|Stage2-B PROTO、DIAGOFF、FULL独立旧类表|factory；executor；Stage2-B scorer；state plan|verified|300-row state plan、旧类registry、same-row旧类评分单测|不得借用Stage2-C新类指标|
|P2-TR-03|7.2|七个同权限主基线|factory；executor|verified|七臂配置diff及23臂数值可达测试|adapter-head已接真实12-step support-only低秩adapter|
|P2-TR-04|7.3|P2-A0联合特征整体消融|executor；factory|verified|identity160-only数值执行与全类预测|真实checkpoint smoke仍待|
|P2-TR-05|7.4|P2-B0稳健中心整体消融|executor；factory|verified|普通support均值、不读地面谱路径可达|真实checkpoint smoke仍待|
|P2-TR-06|7.5|P2-C3旧/新任务均衡消融|executor；factory|verified|D81型全部类等权路径可达|真实checkpoint smoke仍待|
|P2-TR-07|7.6|P2-D0/D1/D2双几何与融合消融|executor；factory|verified|full-only、block-only、固定0.5/0.5三臂可达|真实checkpoint smoke仍待|
|P2-TR-08|7.7|P2-E0关闭Fisher residual|executor；factory|verified|D46无Fisher路径可达；FULL保留D62安全门|真实checkpoint smoke仍待|
|P2-TR-09|7.8|P2-F0/F1/F2/F3量化对照|quantization；executor；resource receipt|verified|state bytes、logit误差、flip、延迟字段及无FP32 sidecar测试|F3为P2-FULL物理别名；无整数kernel只声明存储压缩|
|P2-TR-10|7.9、9.1|K1/K2精确fallback|D42/D43/D81/D92；executor|verified|K2边界、单位协方差、全类argmax单测|真实artifact逐logit闭合待正式row|
|P2-TR-11|4.2、9.2|75-row/arm screening|spec；plan；sealer；matrix runner|implemented|计划生成、物理别名去重、16槽dry-run测试|完整T1为19个逻辑arm，其中F3与FULL共用物理执行；不是全矩阵总共75row|
|P2-TR-12|4.2、14|fresh confirmation覆盖5receiver×3scene×5seed×≥3draw|confirmation plan/launcher|pending|待screening后冻结核心arm|不得用screening结果挑receiver、scene或draw|
|P2-TR-13|11|每row完整artifact字段|row executor；truth scorer；terminal；runner summary|implemented|prediction、behavior、quantization、resource、score、terminal正负测试|正式run还需验证所有logical score数等于计划数|
|P2-TR-14|5.3、9.1、11|immutable prediction先发布，truth-side scorer后连接|prediction artifact；truth scorer；score CLI|verified|预测验证先于truth open、别名评分、不可覆盖测试|runner不读取性能值|
|P2-TR-15|2.2、9.1|query不可达、全注册类逐样本argmax|feature builder/cache；executor API；protocol negative tests|verified|fit API无query、cache无truth、逐样本全类argmax|真实checkpoint no-query smoke待P1-FULL闭合|
|P2-TR-16|9.1、13|真实row executor与冻结full-matrix executor|row CLI；plan sealer；matrix runner|implemented|16槽、外部占用等待、复用、别名、异常指纹dry-run/单测|独立复审进行中；真实checkpoint smoke后才seal正式run|
|P2-TR-17|9.1、13|参数量匹配、配置单因素diff|factory tests；executor receipts|verified|23臂全部数值可达；单因素配置diff；adapter参数/epoch资源闭合|参数量是记录项，不把不同方法强制成同一算法|
|P2-TR-18|13|连续session持久状态与原子rollback接口|后续lifecycle runner|deferred|T3前验证|不阻挡T1 screening发布|
|P2-TR-19|13|目标硬件整数kernel|后续资源实现|deferred|T3前验证|无kernel时只声明存储压缩|
|P2-TR-20|8.1、14|Phase1×Phase2最小2×2|后续2×2 plan/launcher|deferred|T1 Phase2闭合后验证|每cell使用自身Phase1 bundle|
|P2-TR-21|7.10、14|5+5+5连续注册与3种到达顺序|后续lifecycle plan/launcher|deferred|T3前验证|顺序必须在query前锁定|
|P2-TR-22|5、10、14|same-row汇总、paired CI、per-receiver/per-class|`summarize_full_ablation_phase2.py`|implemented|现有测试待纳入发布回归|需确认完整矩阵输入契约|
|P2-TR-23|9.1、13|独立复审P0=0/P1=0、Git提交、不可覆盖N607发布|review artifact；report；bundle/seal|implemented|独立实现审查P0=0、P1=0；244 passed、2 skipped、0 failed|审查时唯一发布P0为未Git封存；本报告所在提交闭合后归零，正式seal仍等待P1-FULL输入|
|P2-TR-24|3.1、9.1、11|把T1完整P1-FULL checkpoint与同row prototype编译成可加载的不可变Phase1 deployment bundle|`build_full_ablation_phase1_deployment_bundle.py`；测试；Phase2报告|implemented|真实P1-FULL seed7281105 checkpoint在batch 1/8/64/256逐项parity；46项回归与独立复审P0=0、P1=0|N607正式签名往返仍在进行；必须输出TorchScript、160维多域聚合组件、class-handle binding、外部签名和正式binding，不得复制source样本或把裸pth带入Phase2|
|P2-TR-25|2.2、4.2、9.1|已有LEO_weak cache按可用row复用，不要求不同启动批次数据一致；每个正式row的support/query/new-class-draw种子分别生效|predictor bundle builder；Stage2输入registry；feature-cache builder；release plan|implemented|D18 cache可复用；support/query seed独立生效；Stage2-C完整target-new pool与draw seed强制闭合；待N607生成正式package、truth-sidecar和registry|仅核对存在、schema、row内部绑定和`VALIDATED_ONCE`句柄，不重做原始数据审计或跨批次对齐；不得把一个总seed伪装成三个独立seed|
|P2-TR-26|9.1、11、13|把每个逻辑row精确绑定到当前已封存feature cache、predictor package和truth-side scoring manifest|`build_full_ablation_stage2_binding_registry.py`；测试；release plan|verified|exact identity映射、artifact链复核、全集覆盖、缺行拒绝和P2-F3/FULL别名共享测试；Stage2-A四个零support字段及严格truth schema/stage/receiver负测已补；最终增量复审P0=0、P1=0|registry只绑定当前启动实际采用的缓存，不要求不同启动的数据或缓存hash相同；待N607真实artifact闭合|
|P2-TR-27|3.1、9.1、11|Phase1 checkpoint→TorchScript parity必须在N607 CUDA上按固定数值策略闭合，并拒绝会改变6类决策的runtime|deployment builder；parity receipt；bundle loader；测试|implemented|旧release CUDA诊断最大绝对差`0.0009131431579589844`、相对标度≤`7.1414e-05`、全部batch在`atol=1e-3,rtol=1e-4`下allclose、argmax mismatch=0；formal prepare入口在任何input/hash/checkpoint访问前强制CUDA与`CUBLAS_WORKSPACE_CONFIG=:4096:8`，runtime层二次检查；receipt/loader精确绑定新版数值策略；7文件64项通过，最终独立复审P0=0、P1=0|v1/v2 partial目录永久保留为`NO_STAGE2_RUN / NO_PERFORMANCE_RESULT`；当前只差新commit和fresh v3 N607 CUDA receipt，正式命令必须由父进程在Python启动前注入`:4096:8`|
|P2-TR-28|3.1、9.1、11|before/after predictor package的旧6类handle必须与formal Phase1 deployment固定class binding逐项同序，禁止每次封包随机重建旧类handle|predictor bundle builder；formal deployment binding；current-launch class-label binding；feature builder跨链测试|verified|启动前覆盖率检查发现旧实现125/125会在首个feature cache稳定失败；首轮复审P0=0、P1=2后，已用current-launch attestation把当前checkpoint lineage、semantic handle binding和formal deployment digest原子绑定，保留旧TX→handle映射复用证据且明确不要求跨启动数据一致；artifact路径、digest、authority和formal loader参数传播均有cache-open前负测；独立实测3文件42项、7文件79项通过，`git diff --check`通过；第二轮独立复审P0=0、P1=0|允许Git封存并按新commit重生成计划和新run ID；旧6fd source plan不再允许启动|
|P2-TR-29|4.2、9.1、11|已一次准入的旧非external LEO_weak cache若仅缺少后来新增的`overlay_role_policy`声明，可在逐行证明全体`channel_views=rx_base`且`overlay_applied=true`后继续复用；任何显式值和external模式仍按当前合同精确要求|`leo_weak_cache.py`；D18真实manifest证据；兼容正负测试|implemented|25c首次package批次50/50以同一指纹在cache loader失败并按规则停止，0 package/feature/prediction；真实三场景均仅缺字段且逐行证据满足`all_roles`；首轮复审P1指出显式null与external缺键边界后已收紧，显式错误/null、external缺键和任一未overlay行均fail-closed；单文件16项、5文件75项通过|待第二轮独立复审、Git封存、new input/run ID和fresh D18 real-cache no-query smoke；不得远端补写旧manifest或原地重启|
|P2-TR-30|3.2、7.4、9.1、11|正式Phase1 v2 center-lowrank-residual-radius组件必须通过outer joint seal authority进入feature builder，并生成D89半径可靠性D81型类无关扰动谱；不得回落到D66 v1 dense组件|`stage2_ablation_feature_builder.py`；formal deployment loader；D89 core；测试；release report|implemented|v5真实D18 smoke的2个package闭合后，feature在旧D81→D80→D66链以同一v1成员缺失指纹停止，0 feature/prediction；修复严格绑定sealed package/component目录、manifest SHA与外层已验证component对象，错误目录/hash负测fail-closed；8文件83项通过|fresh N607 smoke必须改用`artifacts/phase1_unsigned/package/component`，先重新加载Stage2-A/B/C三份cache和audit，再允许全批次；旧v5不得续跑|
|P2-TR-31|3.2、9.1、11|N607正式prototype转换不得依赖Torch2.1/NumPy2不兼容的Tensor`.numpy()`ABI；6×160封存prototype须保持值、顺序、归一化和288维padding语义|`stage2_ablation_feature_builder.py`；feature builder测试；v7 CPU preflight证据|implemented|v7中formal runtime与ground spectrum均PASS，prototype探针在`.cpu().numpy()`边界稳定SIGSEGV且0GPU/0输入输出；改为`cpu().tolist()`后显式构造float32 NumPy，原数值测试与新增禁止ABI bridge负测纳入九文件117项PASS|待独立复审、Git封存和fresh v8 N607 CPU-only三段预检；不得续跑v7|

## 当前最高风险

当前最高风险已经从“无真实输入”转为“N607正式artifact封存尚未闭合”：`P1-FULL__train_seed_7281105`的完整checkpoint/prototype/terminal/receipt已通过本地真实runtime smoke，Git发布提交为`fff5cad186d40ed25335d2095ed7b4007a6651be`，Phase1 deployment bundle正在N607进行不可覆盖落地和外部签名往返。代码侧已具备一次性特征提取、缓存复用、逻辑row全集绑定、物理别名去重、不可覆盖预测/评分、16槽调度和系统性技术故障止损。不同发布批次可以使用不同既有合法数据包，只要求各自row内部预测、truth-side评分和输入身份闭合。

## 数据复用边界

- 已有`phase2_data_status=VALIDATED_ONCE`且`capsule_id/split_id/protocol_schema`匹配的切片直接复用。
- 候选、arm、checkpoint、方法状态、资源预算和发布批次变化不触发数据重验。
- 不要求不同启动批次绑定完全相同的数据缓存；每个row只需绑定自己的既有合法缓存，报告保留来源。
- 已有完整不可变预测可以用`reuse_prediction`复用，并继续生成当前logical row的独立same-row评分；不完整预测不得复用。
- 仅在received IQ、物理ID、receiver/TX集合、场景、K、support/query划分或schema变化时处理对应数据项。
