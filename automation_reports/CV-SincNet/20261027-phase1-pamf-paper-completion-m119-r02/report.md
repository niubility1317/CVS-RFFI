# PAMF 论文补齐：三视图完整SSL、增强PL四格与两层结构消融

- run_id：`20261027-phase1-pamf-paper-completion-m119-r02`
- group_id：`phase1-pamf-paper-final-completion`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Full eleven-setting completion; 119 scratch classifiers, 81 exact/equivalent frozen reuses; no permuted ordinary PL; no optional internal SSL extension; no new DURA.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 最终矩阵与复用边界

用户已明确要求再次全面检查后启动。执行完整 11 设置，保留纯 CE 与完整 SSL 两层结构消融；不执行 PAMF-PL (Permuted)，可选 12 个内部 SSL 机制移除也不纳入本轮。

本 run 共 200 个逻辑 F：119 新训练（111 个 E50、8 个 E200），81 个既有 F。75 个旧 F 的完整预测/独立评分直接复用，6 个三/单视图重排 F 复用既有 raw 预测，只补缺失读出。新阶段共 369=119 训练＋125 预测＋125 独立评分；0 新 G，0 新 IQ 缓存。

|补充内容|新 F|设置|
|---|---:|---|
|完整 Tri-SIG-TIME SSL 主比较|30|Main已有，其他10设置×3；三视图、固定重排、完整U机制|
|无增强监督/普通PL控制|66|11设置×2方法×3；关闭As变换，保持真实标签前向剂量；普通PL0.95/权重1|
|完整SSL结构消融补缺|12|去时间坐标、去PA、去参考、参考相位扰动；均保留同一固定重排，各3种子|
|同排列监督权限控制|3|只用L；用于识别器Sup/SSL与G-L/G-LU的四格|
|三视图纯CE架构比较|8|E200/10000，L6300、无U、无增强、自然序；与已有单视图E200配对|

旧CE九结构27模型和去SIG五组新增12模型，共39独立历史模型全部保留，不重训。完整SSL九结构的其余五项复用既有完整三视图、单视图、自然序REPL、无SIG和矩统计替代。

## 数据、方法与测试

沿用 Main split、Receiver split1/2、1%/5% labeled、1/2/3/4/6 source receivers、10-class/16-source设置。10%和共同六目标曲线的5源点均复用Main；不新增训练契约。L/U/V及接收机具体集合见[总设计](../../../docs/reports/pamf_paper_master_design_20261027/report.md)。U的TX标签始终为隐藏值，V不参与梯度/状态更新。

E50保留20+10+20=11100更新及原联合默认头。Sup不读取U；普通PL在E14启用，阈值0.95/权重1；完整SSL在E14–20保留普通PL、E21–50启用原完整REPL权重6、风格扰动、梯度保护、joint160对比和接收机外推。无EMA。完整SSL采用固定窗口排列[0,5,3,6,2,1,4,7]。当前普通PL保留自然序，所以完整SSL对PL的差异解释为方法组合收益。

全部E50行（包括关闭训练信道增强的行）自动评测clean和六种practical；跨日、跨RX、双跨分别给raw/DURA-L/DURA-LU。七景决策全部固定后才连接独立truth-side评分。D0更早，不声称未来日期外推；168000综合与198000综合分开。E200只测clean，给168000综合、126000跨RX及42000双跨。

DURA使用此前22个匹配缓存，逐设置保留L/LU权限；现有21组八阶段权重和Main原版谱系/物理ID核验复用，未发现六源G混入五源F。G跨F种子共享，不声称恢复器三种子独立重复。

## 发布前检查

只读再次核实75个完整读出、22个G缓存；E200的168000物理ID严格包含当前42000双跨ID。两份历史评分位于原技术恢复目录，按最新jobs解析后直接复用，无重测。

10项聚焦协议负测通过；冻结依赖环境的三视图CE图232993参数、有限2x6输出通过。唯一独立P0/P1审查通过；发布包缺comparison_suite的问题已修复并做一次定点确认，4个新增结构组合的实际CPU前向/反向通过。r01在GPU启动前因两个未参与计算的历史记录字段误比较而退出；修复后仍比较8项实际机制，并检查真实6660次完整SSL更新及20+10+20段，9个既有源实证通过，新增回归和一次定点复核通过。旧目录全部保留，r02不改变科学矩阵。真实已训练checkpoint无query smoke是launcher第一步，PASS后立即自动进入矩阵。

代码与配置由发布时Git HEAD固定；实际commit写入release_commit.txt和runtime.code_commit，启动核验后补本报告。资源为GPU0–7，每卡最多一个本run GPU worker、总实验进程最多2、空闲显存至少12000MiB；任何其他UID占用的卡跳过。全局最多一个独立CPU scorer。保留其他owner的健康进程。

首个协议/来源/输出冲突、非有限数、无法执行或不能产生合法prediction的技术失败停止新派发，健康已启动行保留完成，产物全部保留；低成绩不触发停止、调参或选择性重跑。所有行冻结后自动测试、独立评分并生成完整种子统计。

当前状态为已登记、待发布；不能将本地验证或提交动作当作远端训练已启动。
