# CORE90设计复查与seed392005发布记录

用户要求：再次对照原设计、计划书检查实现及未启用机制，验收后发布；本轮seed392005，允许超过每GPU2个实验。保留前轮健康任务，不复用其checkpoint。

## 编辑前追踪

|ID|来源|要求与实际缺口|文件|状态|验收|
|---|---|---|---|---|---|
|A01|原报告10.4；计划R02/R26|U池按epoch顺序重置，约8/9未消费；需保留时间邻接的可恢复全池覆盖|data.py、runtime.py|pending|跨epoch覆盖、同批邻接、恢复一致|
|A02|原报告3.3；计划R08|每U batch清Optimistic历史，pseudo阶段全部退化；按显著问题变化重置|runtime.py、solvers.py|pending|稳定伪标签分布使用历史，大变化重置，checkpoint恢复|
|A03|原报告9；计划R14/R16/R23|首轮未派发B3/B7/B8/S2/C3/C4/H1，Jacobian默认关闭|matrix/release/dispatcher|pending|seed392005完整适用菜单、专用高阶行、依赖日程后派发|
|A04|计划R21/R23|同动作计数对照只能来自完整其他seed donor；不能消费本seed未来信息|replay/matrix|pending|完整source donor及不同seed，固定/随机/重放计数核对|
|A05|用户本轮覆盖|允许超过每GPU2进程，必须按实际显存/负载调度|dispatcher/release|pending|可配置容量、显存准入、GPU UUID、启动占位及独立读回|
|A06|全部|原26项逆向核对：实现、配置、自然触发、阶段等待和刻意消融分别报告|本报告及完整日志分析|pending|原始报告/计划/参数/完整已有日志/新run激活记录|

原报告融合是条件接口，CORE90的z_dom不是身份响应分支，仍不强行新增融合。隐式响应为已披露最终线性层条件近似。新实验不强制把未达到合法信号条件的控制动作触发。

## 复查结论与修复

确有两项会改变后期实验行为的缺陷，已修复。A01：U训练按连续窗口打乱次序并跨epoch轮转；56700条U、batch128、每epoch49步时10个伪标签epoch覆盖全池，恢复由seed和epoch确定，既不读取隐藏TX也保留窗口内时间邻接。末尾单样本并入前一窗口。A02：删除每个伪标签batch无条件清空乐观历史的逻辑；改为20步观测窗的选中比例或类别分布TV变化超过0.25时重置，观测累积状态纳入checkpoint。阶段、课程和目标权重变化仍重置历史。

A01/A02已完成定点回归及独立P0/P1复核。真实训练入口的合成伪标签验收共6次更新，全部接受，后5次实际使用Optimistic历史，6次均未发生不必要重置。此测试提前开启伪标签仅为功能验收，正式E131阶段不变。

A03已配置26行seed392005、E200、scratch-only矩阵，新增/补发Jacobian专用J1(interval500)、条件隐式H1、B7、B8、B3调优/头尺度、S2/S2_random及C3；高级机制优先派发。A05改为本轮每GPU最多4个计算进程，计入已有进程，显存准入4096MiB、启动占位3072MiB，继续使用GPU UUID绑定。旧队列和健康进程保留。

A04尚未满足发布依赖：严格跨seed C4以及匹配动作计数的固定/随机对照需要修复后完整的其他seed donor。已向用户询问是否额外授权一条seed392002 donor；在未收到授权时全部新训练保持392005，C4不发布。现有S2/S2_random/C3是预设比例机制对照，不冒充严格匹配实际成本。旧seed日志受A01缺陷影响，不作为合规修复版donor。

## 原26项逆向核对

既有逐项证据见[实施验收报告](CORE90_GAME_IMPLEMENTATION_ACCEPTANCE_20260911.md)。本次对R01—R26全部复核：23项功能实现已验证；R18为条件融合接口不适用；R21/R23的跨seed严格成本对照为2项部分完成、依赖donor。R16虽计入功能实现，明确仅最终线性域头条件近似，不能称整个非线性头精确响应。R14限制末端身份块与域头，不能称全网Jacobian。R08以本次20步窗口修复取代前报告的“U阶段保守逐batch重置”。R02增加U全池轮转；R26本轮容量覆盖为4。

功能验收不等于自然激活或研究验收。R12独立探针、R14高阶审计、R16补偿、R19课程和R20控制都必须由正式日志分别确认。E80卫星与E131伪标签尚未到阶段属于等待；零次控制动作可能是未满足合法信号，不能强行触发。全部正式E200、四场景封存预测、独立评分和多seed科学比较仍待完成。

## 上一轮完整日志快照

完整解析快照内全部11个已启动run的JSONL，运行中下载的tar有“file changed as read”，因此这是时间点快照，不是最终完整训练。B0/B1/B2每行约595—610次主更新，B4两行323/322次主更新、646/644次头更新，B5两行394/380次主更新；B1/B2均已有3次合法source审计并生成校准。没有E200结论。原22行菜单未包含B7/B8/H1/J1，未派发不能记为机制运行成功。本轮不重启、不覆盖旧产物，旧结果需带U采样缺陷说明。

新26行矩阵：J1、H1、B7、B8、C2、S4、S3、S1、C1、B0、B1、B2、B3、B3_lr_high、B3_head_slow、B3_head_fast、B3_head_scale、B3_adv_low、B3_adv_high、B4、B4_fixedk、B5、B6、S2、S2_random、C3。训练seed均392005，固定split seed392002不属于训练seed或checkpoint复用。

最终本地相关测试107项全部通过，覆盖activation、analysis、audit/control、compat、field、integration、matrix、reaudit、replay、resume、solvers；只有既有AMP弃用警告。独立定点审查无新P0/P1。发布后的进程与实际激活证据另附，不以本地功能测试替代远端观察。

## N607发布与实际机制读回：VERIFIED

训练代码提交`300ff4a9f3789a846312f00c19b44c9dde1f54fa`已推送，独立读取GitHub分支OID一致。归档SHA256=`efe1c3b27571f791aa914db98327d8bbf043b00b0ae7be2373cc0199086ab5c6`，远端核对后新建release。没有复用前轮checkpoint。

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/core90_game_392005_300ff4a9`。
- run-root：`/home/szu2070436088/2510044040/CV-SincNet/runs/core90_game_392005_300ff4a9`；dispatcher PID1329605。
- 远端Torch2.1.0+cu121真实源数据scratch checkpoint重载和optimizer更新PASS，query_access=false。
- 两次进程快照确认16行运行、10行排队、0失败；16行CWD、PPID、完整argv、GPU UUID、seed392005、E200、from_scratch=true、空baseline/resume一致，日志实际增长。首快照8卡各4进程；第二快照GPU2增为5个，其他7卡各4个。独立ps确认新增PID1343028属于另一个`core90_evidence_frozen_392005_20260911/H0`任务，PPID1342406，不属于本dispatcher1329605；未干预。GPU2仍有14548MiB空闲。本队列容量4限制其派发，不能约束其他队列事后启动；本轮及已有PID均保持存在。
- 正式日志快照已到E1—E2，16行84—124次更新均接受。B7的124次更新中123次使用乐观历史；B8的84次更新均执行两次场求值。
- J1实际执行完整当前有效目标的局部有符号场审计，valid=true；按预登记缩步至6.25e-5消除ReLU跨区，有限差分相对误差0.0771318<0.1。
- H1条件最终线性层补偿实际执行7次HVP，残差2.79e-7，source monitor CE从2.69067降至2.67351，accepted=true。该数值仅说明局部补偿接受，不是研究性能提升。
- 独立读取正式B0的E2 checkpoint：step98、scratch_only、final_only、target_contact=false；L/U/V=6300/56700/27000，source RX=`1,3,4,6,8`、days=`1,2,3`、6个TX、15个RX×day域，契约符合本轮设置。

[启动证据](evidence/core90_game_392005_launch.json)、[进程与增长证据](evidence/core90_game_392005_progress.json)、[完整当前动作日志汇总及机制/checkpoint证据](evidence/core90_game_392005_activation.json)、[上一轮完整日志快照分析](evidence/core90_game_previous_snapshot_392005.md)。

本次结论为：已修复两个实际缺陷，相关功能验收通过，26行修复版实验已发布，部分高级机制已实际执行。A01/A02功能验证完成；A03/A05发布验证完成；A06逐项核对完成；A04跨seed严格对照仍待donor授权及完成。E80/E131机制、自然控制触发、E200预测评分尚在后续阶段，不宣称已全部激活或获得性能收益。
