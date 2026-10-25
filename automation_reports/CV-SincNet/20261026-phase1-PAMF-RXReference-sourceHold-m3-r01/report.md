# 跨接收机身份参照三变体单种子源留出验证

- run_id：`20261026-phase1-PAMF-RXReference-sourceHold-m3-r01`
- group_id：`phase1-pamf-receiver-reference-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（完整单种子源矩阵完成，三个候选均未达扩展条件）

## 目的与对照

同seed01原E30+20至E50，raw/同G/双G身份参照三变体；新增分支排除RX1，完整源V八视图与既有PLControl配对，源阈值选择后另行扩种子。

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

## 发布前范围

详见 [最小预登记](preregistration.md)。三个方法变体仅有一个student seed2026092701；不是三种子测试。固定原E30+20、共同源剂量0.07145844915553692及默认头联合训练，完整源V八视图逐行自动评估；源条件通过后未来三种子矩阵另行登记。本run不访问目标数据，不替换默认模型。两项新检查与compile通过，一次独立P0/P1审查PASS；真实权重联合backward smoke为launcher第一步。

## 三条源验证训练已实际更新

VERIFIED_RUNNING，科学代码 `37dc44e3dd52854a415ca5f7663669b417171691`。三个变体均已进入真实更新，实际物理GPU为0/0/1，只有student seed2026092701。真实E30/defaulthead/L128-U256联合backward smoke通过；独立PID/CWD/argv/GPU和step日志增长核实。当前可用完整新增step记录分别为[261, 262, 278]，不是E50完成或新目标结果。原L768包、AsKL0、隐藏U、共同升温权重和零G/参照更新的最新记录符合配置。见 [启动证据](launch_evidence.json) 和 [启动源进度](startup_source_progress.csv)。逐行训练后自动完成完整源V八视图；本run不包含目标访问或三种子扩展。

## 运行阶段完整日志前缀核查

独立检查实际initial checkpoint及全部当前完整step记录，核对原默认头6×160、own E30父链配置、L768/AsKL0/CRIC.1、完整U曝光、排除RX1的新增接受计数、共同权重和非有限/真值标志。三个变体在相同步数的新增接受流完全一致，每完整epoch新增接受41939条；全U仍进入原raw PL主项。G与参照未更新，无新目标读取。

完整可用记录数、观测UTC时间与每行进度见 [阶段源执行](available_source_execution.csv) 和 [阶段证据](available_source_evidence.json)，快照可随观察更新。已缓冲而未形成完整JSON行的尾部不计入；这不是全部E50完成或目标收益。loss仅作源训练健康观察，不改变已冻结的源选择规则、训练预算或候选顺序。

## 完整单种子源留出结果

`SOURCE_HOLDOUT_ANALYZED`，独立读回VERIFIED。三个方法只有student seed2026092701，均固定own原E30+20至E50；每条完整4440新增更新及最终默认头checkpoint核实，共13320条训练记录。原L768包、As KL0、CRIC.1、全部U曝光与固定新增接受流符合配置，未发现非有限跳步、U真值、G或参照更新。原U每条曝光1134000，留出RX1曝光226800只进入原主项，新增接受838780，额外两前向2268000。

以下相对同seed/同物理V/同场景的原PLControl，单位pp。源V只有原单一27000角色，留出RX1仍是原sourceRX；E30/G和主项见过它，不能当作真正从未见过接收机或目标测试。

| 方法 | 留出RX三个G场景均值差 | 整体源raw clean差 | 预定源条件 |
| --- | ---: | ---: | --- |
| RawReference | +0.067901 | +0.022222 | FAIL |
| RestoredReference | +0.117284 | +0.014815 | FAIL |
| PairedReference | +0.098765 | +0.011111 | FAIL |

源选择：**无候选通过**。规则为G均值差≥+.3pp、源clean差≥−.1pp；通过者按G增益最大选择，精确相同按方法字母顺序。全部三个变体及负结果保留。未来三种子/目标实验须另行冻结矩阵，本run没有执行新目标访问，以上结果不能证明原clean+1pp、raw星地>1.5pp或同G提升目标。

参见[全部源分层](source_all_strata.csv)、[源单种子端点](source_one_seed_endpoints.csv)、[固定源判断](source_candidate_decision.csv)、[完整训练执行](finished_source_execution.csv)和[独立证据](source_result_evidence.json)。评估阶段F/G参数及全部registered buffers不变，默认头直接预测，无另训头；学生在此前20轮中正常更新。

## 源筛选结论与下一步问题

三个候选均未达到预定+.3pp留出G增益，故不扩三种子，也不启动其目标测试；所有模型和结果保留。源clean没有明显损伤，但微增不能替代原目标。训练新增关系loss下降，源默认头识别收益仍小；现有结果尚不能区分保守剂量不足、身份参照目标与默认头不匹配等原因。

| 方法 | 新增分支RX组 | 三个源G场景均值差（pp） |
| --- | --- | ---: |
| RawReference | held | +0.067901 |
| RawReference | trained | +0.032407 |
| RestoredReference | held | +0.117284 |
| RestoredReference | trained | +0.191358 |
| PairedReference | held | +0.098765 |
| PairedReference | trained | +0.174383 |

分组见 [源迁移汇总](source_transfer_summary.csv)。下一步先在合法L/U上冻结模型，测量辅助目标与默认头身份目标的特征梯度关系，再决定是提高源剂量还是更换目标；不因本轮不通过而放宽原阈值。

最终checkpoint只读核验工具首次因未激活既有legacy模块路径而读取失败，补齐与launcher相同的activate后全部读回通过；没有重训、重启或修改任何模型。
