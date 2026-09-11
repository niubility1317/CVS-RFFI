# CORE90博弈优化V2实施与验收记录

当前授权：严格按新设计报告和实施计划实现并确认验收；不派发新的正式训练矩阵。

依据：[实施计划](CORE90_GAME_V2_REPAIR_IMPLEMENTATION_PLAN_20260911.md)与[设计原文](design_refs/CORE90_GAME_V2_DESIGN_REVIEW_20260911.txt)。起点提交9ed1c49e。当前隔离工作树`code/snapshots/core90_game_20260911_wt`；保留无关未提交`report_core90_target_test.py`。

## 编辑前任务与接口核对（历史阶段记录）

|任务|实现责任与接口|状态|
|---|---|---|
|A|audit_evidence/source_audit：v2证据、经验gap、跨TX读出、优化轨迹；runtime负责接入|pending|
|B|data/runtime/controller：均衡参考、独立时钟、质量门控与一次动作；消费A|pending|
|C|negative controls：真实训练上下文、持久状态、零对抗等价；消费A/B和求解器|pending|
|D|solvers及B8专用模块/benchmark：保留reference、head-only梯度、图复用验收；不修改共享runtime接口|pending|
|E|curriculum/capability/step_context/runtime：有效连续policy和当前/下一难度身份测量；消费独立能力证据|pending|
|F/G|矩阵/replay/analysis：配置及独立固定预测分析；不启动正式训练|pending|

A/B共享数据与证据接口由主任务整合；D不直接改step_context/runtime以避免与E冲突，必要接口由主任务接入。F/G使用现有配置字段生成计划，新增字段先与主任务约定。所有任务保持scratch、source-only、E80/E131和原有loss。

实现范围R01—R18与R21的本地部分；R19追加B4/B7修复训练、R20复杂度扩展按计划暂缓。R21正式发布/E200新性能不在本轮授权。图复用若不等价按计划保留已通过的head_grad_only并披露失败，不以近似替代reference。

初始验证：本地ssr-gpu Python为`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，Torch2.10.0+cu128，CUDA可用。最终状态以下文验收表为准。

## 已闭合的功能证据

V2审计已接入真实训练循环，经验优化gap、跨TX域/RX读出、完整方向参照与身份能力分别记录。方向参照涵盖完整TX×RX×day容器；同时区分普通TX CE、完整labeled非对抗目标、含卫星/U的完整当前训练目标。U仅使用原训练上下文的合法伪标签和mask。所有审计使用owned模型副本与隔离随机数，不更新训练BN、EMA、原型或optimizer。

两个独立审计时钟及定标状态进入V2 checkpoint；V1控制状态不能静默恢复。一次可信观测至多申请一次动作，实际head步和已接受EG单独记账。有效卫星policy或目标阶段改变后旧博弈证据失效。能力课程使用同一fit-only身份头读取clean、当前与下一难度的TX/RX×TX风险，具有进入/退出滞回、3次新观测确认、250步冷却及0.1升级上限；余弦和lag不是升级门槛。

实际runtime的有审计/无审计/禁用动作三条轨迹已逐项比较。合成功能回放覆盖E1、E40→41、E79→80、E130→131及伪标签mask全False→全True；epoch中断恢复也已验证。真实WiSig源数据另完成7个阶段上下文、两条相对基线分支，共14个逐步完整state比较，全部`first_difference=null`。真实短程回放使用2步探针，不能代替120步选中探针的全状态对照，也不是从E1实际训练到E131。

跨作者P0/P1审查已修复：V2 head追赶仍用eval-clean、replay绕过recipient质量保护、旧在线fixed/random混入严格对照、H1参考及质量未接线、曝光文件未绑定物理信道配置、阶段切换未失效旧证据，以及嵌套旧capability绕过版本保护。V2追赶现使用训练态完整main+sat特征，CE仅监督main行，只提交head持久状态。一个实际训练入口的受控信号测试已确认依次执行CATCHUP/CORRECT/NORMAL，提交head步为1/0/0、场求值为1/2/1、动作事件2次且无悬空请求。该受控信号测试不证明真实数据必然激活动作。

## 真实source探针开发及不能省略的失败

数据使用本地`local_artifacts/cvs_publication_inputs_20260713/ManySig.pkl`，scratch初始化，无checkpoint加载、target访问或V拟合。实际L/U/V=6300/56700/27000，6TX×5RX×3day共90源容器；经验gap每容器4条，共360条。只把scratch模型置于E131目标上下文，不称为E131训练模型。

五候选在相同初始头、同一固定目标上比较，初始CE均为2.721549；lr={0.0002,0.002}×步数={40,120}为候选，旧0.02/40仅为参考。预定选择规则是完成预算、质量可靠、control-ready后取同目标终点CE最低者；没有按target、方向结果或monitor选择恢复step。

选中0.002/120时，360条经验目标CE降至0.128511，归一化gap约0.95753，状态`RELIABLE_HIGH_GAP`。但同一已选头在90条方向参考上的CE从2.729933升至4.533828，增加1.803895；配对非对抗梯度残差约4.17e-7。该方向参照明确为`TRANSFER_FAILURE/valid=false`，不能用于CORRECT。保留原选择及失败产物，没有重新挑头、改seed或重拟合来消除此失败。高经验gap仅描述固定训练子集，不证明全分布最优反应或可迁移的方向恢复。

冻结的`selected_probe_config.json`已由真实CLI解析为V2、lr=0.002、120步。通用parser保留40步默认以维持既有配置兼容；未来采用本次source开发配置时需显式传入该冻结文件。方向阈值0.5/0.3仍为fixed，不能称所有阈值均自适应定标。

## 配置与实验边界

已生成18行V2_A—F配对配置及3行强普通source候选，均为`CONFIGURED_NOT_LAUNCHED`，scratch、E200、fixed课程、no-audit。强普通候选只改变lr；选择要求三个完整source E200终点、四场景平均Macro-F1最大，平局取较低lr，再把同一所选配置用于全部确认seed。候选没有实际完成时返回DEFERRED。

严格动作回放要求合法独立source donor及匹配契约，recipient仍按当前可信证据和预算接受或拒绝；日志分别保留冻结请求、实际决定和提交数。曝光回放绑定真实mask、场景、信道种子、样本顺序及物理信道配置，区分E80前后计数；count匹配不能冒称exact样本/信道匹配。没有完整donor时不生成已经完成严格对照的结论。

固定预测分析工具已加入TX/RX×TX弱项、混淆、预测直方图与共同clean-correct条件鲁棒性；truth只在声明的预测全集闭合后打开。缺历史类直方图或梯度记录时，B4首次塌缩/B7历史预测能力保持UNKNOWN，不补造根因。R19修复训练、R20复杂度扩展及R21远端正式训练与新评分按本轮边界暂缓。

## 集成检查记录

一次全集成检查覆盖188项：187通过，1项旧V1动作fixture未显式选version1而失败。修正fixture后相关2项通过，包含新增的V2真实动作提交测试；阶段证据失效的新增聚焦测试也通过。保留初次失败记录，不将其改写为首次全绿。现有warning来自旧PyTorch AMP接口弃用提示。

B8最初非确定CUDA路径出现连续轨迹分歧，reference对reference也不稳定；已保存失败产物并完成首差定位。显式确定性后端下，原预算9组全部完成并通过终点等价检查，详见下文；核心配置manifest已更新为`VERIFIED_LOCAL_DETERMINISTIC_FIXED_CONTEXT_AND_TIMED_ENDPOINTS`。该结论没有使用失败运行的部分计时，也未放宽容差。

这里的“B8等价”只指head_grad_only/graph_reuse与现有head-lookahead reference之间的实现关系，不证明head-only预测等同完整EG，更不证明性能增益。原点head-only梯度范数与reference全场范数的scope不同，遥测显式区分；正式更新范数按同一参数角色比较。

代码及21份配置已先独立提交为`5e51cc4544545b51bba26b86818b6bef790a4d4d`并push。首次额外远端读回遇到Schannel握手失败，仅改用Git的OpenSSL后端执行只读`ls-remote`，确认远端OID与本地相同；没有重复提交或重写历史。最终验收文档、成本证据及FP32入口保护以本文件所在后续提交交付，最终OID与远端核验结果见任务交付消息。

<!-- FINAL_ACCEPTANCE_TABLES -->

## B8最终有界验收与成本

确定性CUDA后端下，E1与E131的reference/reference、reference/head_grad_only、reference/graph_reuse各8步全部逐位一致。下表使用同一真实source固定batch=90，按原预算完成每组预热20步、计时100步×3；各repeat的120步终点模型、optimizer、RNG、mask及prototype在计时区域外按预定atol=1e-6、rtol=1e-5核对通过。没有失败后放宽容差。

|阶段|实现|中位秒/步|范围|峰值allocated MiB|峰值reserved MiB|每100步模型forward|求解器场backward|
|---|---|---|---|---|---|---|---|
|E1|reference|1.264744|1.197943—1.389722|643.59|710.00|200|200|
|E1|head_grad_only|1.179021|1.155499—1.179179|640.64|710.00|200|200|
|E1|graph_reuse|0.766708|0.755705—0.769824|640.52|710.00|100|200|
|E80|reference|3.531112|3.091145—3.708622|643.59|712.00|200|200|
|E80|head_grad_only|2.928275|2.891230—3.073936|640.64|712.00|200|200|
|E80|graph_reuse|2.060336|1.936565—2.082198|640.52|712.00|100|200|
|E131|reference|3.895713|3.539948—3.927557|909.44|928.00|400|200|
|E131|head_grad_only|3.106831|2.992918—3.198053|906.49|928.00|400|200|
|E131|graph_reuse|1.924628|1.909467—1.929544|906.38|928.00|200|200|

GPU仍有桌面共享负载，纯算法墙钟加速归因为UNKNOWN；表中为实际测得的分布，不承诺固定加速百分比。模型forward包含E131的U强视图，但计时中自然伪标签选择为0且prototype为空；非零有效U/mask及prototype覆盖来自独立功能测试，不能由该计时推断所有分支已激活。独立CPU真实source的9组API调用审计每步autograd.grad=2、create_graph=True=0、额外显式HVP=0；当前既有FISHR实际为解析logit-gradient proxy，没有内部autograd.grad调用。API次数不等于底层kernel数量。benchmark未启用低频遥测，遥测功能另有验证。

[完整确定性计时与终点核对](evidence/core90_v2/b8_benchmark_cuda_deterministic.json)；[最初非确定性首差定位](evidence/core90_v2/b8_cuda_determinism_diagnosis.md)；[CPU及B8实现验证](evidence/core90_v2/b8_report.md)。

## 最终设计追踪

|ID|本轮状态|实际闭合范围|
|---|---|---|
|R01|verified_local|scratch、source契约及版本化恢复；没有目标推理或新正式训练|
|R02|verified_local|恢复失败/预算不足与可靠低gap分离，缺失不填0|
|R03|verified_local|360条固定训练目标经验gap，完整head模式/尺度回放；外推范围受限|
|R04|verified_local|跨TX交换两折、15域/5RX的linear/MLP读出独立记录|
|R05|verified_local|90容器方向参考及H1两侧45容器，拒绝前32条截断|
|R06|verified_local|完整fit轨迹、尺度、优化器冷重置、固定终点及停止原因|
|R07|verified_local|坏lag或方向迁移失败不授权CORRECT，不进入健康稀疏化|
|R08|verified_local|独立clock/定标及一次观测一次事件，恢复与拒绝记账|
|R09|verified_local|合成阶段/mask边界和真实source14对无动作完整状态一致|
|R10|verified_local|关闭全部head贡献后的零对抗退化与非头状态检查|
|R11|verified_local|CPU与确定性CUDA的reference/D1/graph有界等价及终点核对|
|R12|verified_local|裁剪、AdamW位移、角色遥测及完整固定预算计时；共享墙钟归因未知|
|R13|verified_local|连续实际policy、真实mask/seed/计数与仅有效变化重置|
|R14|verified_local|fit-only同head三视图身份风险、独立课程clock和滞回|
|R15|verified_local|实际曝光文件生成/消费/信道契约检查；真实donor实验尚未执行|
|R16|verified_local|18行因子配置、3个强普通source候选及一次source选择工具，均未派发|
|R17|verified_local|独立donor、state_checked请求/接受/拒绝与预算；真实严格对照待donor|
|R18|verified_local|固定预测先闭合后truth的TX/RX弱项、混淆和条件鲁棒性工具|
|R19|deferred|B4/B7新增修复训练暂缓；缺历史统计不能制造根因|
|R20|deferred|H1/J1复杂度扩展暂缓，保留共同测量修复与条件近似|
|R21|verified_local|本地集成与Git交付；远端新release、E200和新固定评分不在本轮执行范围|

计数：verified_local=19、deferred=2、rejected=0、blocked=0。verified_local只表示本轮实现与有界验收范围；不包含表中明确尚未运行的正式实验。

193个唯一测试用例均有通过记录：初次188项中187通过、1个旧V1 fixture失败；修复后相关2项通过，新增阶段失效1项、确定性配置2项和graph AMP拒绝1项通过。该统计是去重后的分阶段验证记录，并非声称一次命令193项全绿。graph_reuse仅验收FP32，配置和求解器入口均拒绝AMP/CPU或CUDA autocast组合。

[原始全集成记录](evidence/core90_v2/pytest_integration.xml)；[去重验证记录](evidence/core90_v2/verification_record.json)；[source候选与迁移失败](evidence/core90_v2/source_probe_development_guarded/development.json)；[真实source负对照](evidence/core90_v2/source_negative_controls/acceptance.json)；[控制审查](evidence/core90_v2/control_review.md)；[能力审查](evidence/core90_v2/capability_review.md)。
