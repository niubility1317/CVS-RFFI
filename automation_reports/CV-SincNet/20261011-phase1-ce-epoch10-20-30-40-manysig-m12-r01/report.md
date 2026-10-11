# CE＋旧门控＋cos：E30/E20/E40/E10各三种子压缩

- run_id：`20261011-phase1-ce-epoch10-20-30-40-manysig-m12-r01`
- group_id：`cvs-ce-epoch-compression`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

用户要求E30/E20/E40各三行，随后追加E10三行。本次共十二行，只改变从零训练轮数与完整cosine日程；全部按最终轮冻结并自动clean测试。

三个模型seed固定为2026092701、2026092702、2026092703。每epoch222次更新：E10为2220次，E20为4440次，E30为6660次，E40为8880次。batch128、峰值LR2e-4、最低LR1e-6、weight decay1e-4，AdamW和原身份网络/内部门控保持不变。

合法源域L的clean CE，label_smoothing=0；实际关闭U迭代、半监督、EMA教师、星地增强和辅助分支。L/U/V物理角色保持0.07/0.63/0.30。源V仅健康观察，不挑选轮次。全行从零训练，不继承E25/E200或其他权重。

每行训练后自动冻结最终轮，预测相同168000条clean query，prediction固定后由独立scorer连接truth。只测clean，不增加practical场景；延续用户已明确授权的epoch开发比较，测试反馈只用于训练轮数研究，不扩展到机制选模或独立盲测声明。每预算齐三个seed后与固定E200原预测比较，均值下降不超过0.3个百分点为描述性容差；不声称统计非劣效，不自动启动新预算。

新输出root与唯一owner；每GPU最多两个总CUDA/预留进程，至少12000MB空闲。技术异常保留产物、停止pending，健康行继续，不自动重试或覆盖。

## 发布前验证

本地ssr-gpu检查覆盖四种预算与cosine端点、十二行输出唯一性、原架构初始state一致、纯CE真实前后向、222步迭代，以及target/resume/预算/视图越界负测。机器证据见evidence/local_checks.json。

独立只读P0/P1审查未发现阻断问题，核对12行完成计数、多预算原预测按seed去重、登记配置一致及发布依赖。审查证据见evidence/independent_review.json。

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

## 实际启动读回

2026-10-11T22:03:55.912105（北京时间），12/12行已启动，pending=0，failure=0。所有PID/CWD/argv/GPU与实际resolved config已核实；全部scratch、无继承权重，实际预算分别为10/20/30/40轮。最早9行已完成第1轮，后启动3行进入第1轮，日志在增长。

|预算|seed 01 的 GPU|seed 02 的 GPU|seed 03 的 GPU|
|---|---:|---:|---:|
|E10|6|2|3|
|E20|1|7|0|
|E30|0|7|1|
|E40|2|6|5|

执行提交 `83b1903fed01854f5abb75646a02e9a5e190a90e`，dispatcher PID2637。验证状态VERIFIED，运行与测试尚未完成。完整独立证据见[evidence/launch_readback.json](evidence/launch_readback.json)。

远端日志目录：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261011-phase1-ce-epoch10-20-30-40-manysig-m12-r01/logs/`。每行自动训练→固定最终轮→clean预测→独立truth-last评分；不需要再次发测试指令。

