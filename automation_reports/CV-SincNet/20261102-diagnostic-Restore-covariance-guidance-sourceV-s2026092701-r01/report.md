# 冻结源 V：物理恢复与特征指导的真实识别收益

- run_id：`20261102-diagnostic-Restore-covariance-guidance-sourceV-s2026092701-r01`
- group_id：`restoration-covariance-feature-sourceV-mechanism`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同原1800源V七景，固定BF物理版与5359特征版final30，保存IQ/160特征/默认头输出；全部预测固定后独立CPU才读V TX，分析特征改善但识别损伤的病例。

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


本项只作原源 V 冻结验证，不训练、选模或调参。固定纯物理 BF 与特征指导 5359 的最终30轮、同一原3ff8完整地面Tri-SIG-TIME SSL，原R07的1800 V×七景，5 RX×3 day每格120；V与训练物理ID互斥，既有验证面板已曝光，不冒充新盲测。

实际Ls2检查两版最终G和原F默认头，随后完成全部七景预测；保存IQ、160维特征、默认头logits和所有逐例指标。全部文件完整后独立CUDA空CPU进程才按原合同ID连接V_metadata的TX标签，不读U真值。报告真实准确率及净纠错、clean参考判决保持、.01(cos+源方差MSE)、clean参考类别margin，以及全Action相对MSE和标量噪声误差。预先固定特征变化容差1e-7，统计特征更近但TX判错的病例。所有正负结果均保留，预测与评分不改变权重、轮次或Main结果。

源V接收机实例seed2026101902，原训练实例2026101911；因此不能将本验证与Main差异全部归因于未知RX。Fclean一致率、margin和特征拟合均不等于真实TX准确率，源V结果不是信息论极限。0OE/novel/target/训练更新；默认DURA不变。矩阵、精确命令、环境/CWD、不可覆盖输出、GPU与停止规则见experiment.json。
