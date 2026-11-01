# 最终双分支恢复器：源V波形与真实识别验证

- run_id：`20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01`
- group_id：`restoration-final-detail-sourceV-IQ-feature-TX-validation`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定1fcb最终30/5481223 G，原1800V七景只新增detail输出；旧received/BF/guided控制与clean特征全部复用。完整预测后独立CPU评分TX，分开报告IQ/特征/Fclean与真实准确率。

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


本项只作最终冻结后的原源V验证，排在父实验完整Main和随后延迟测量全部技术完成之后，不与它们争GPU，不读取其成绩控制开始。固定1fcb最终30/5481223 G与同原3ff8完整Tri-SIG-TIME SSL，原R07的1800源V×七景（5RX×3day×120）。V只验证，0训练/选模/校准/目标访问。

只新增12600条detailG/F0输出，另真实Ls2完整G/F0检查；旧4178源V的received/BF/5359控制IQ、特征、logits和1800cleanF0参考按相同ID与F0身份复用，0新基线神经前向。先固定四方法全七景输出，再独立CUDA空CPU进程才按原V_metadata映射TX评分。分别报告真实准确率、修复/损伤、IQNMSE、连续特征误差、Fclean参考margin以及物理参数误差，特征改善/持平/变差容差固定1e-7，条件比较为detail相对physical。

源V已有研发曝光，同属五源RX，验证接收机实例seed2026101902、训练实例2026101911；不能单独证明Main的域原因、未知设备保真或理论上限。Main未缓存全量restoredIQ，所以本源V波形误差不能改标Main波形恢复率，IQNMSE也不是“发射机信息恢复几成”。这项诊断补充波形与识别收益的对应关系，不替代原Main六景3pp＋两个clean保持。

3新检查与4原检查、唯一P0/P1通过；实际source2为worker第一NN步骤，PASS继续全部矩阵。严格final30/config/state、冻结状态保持、缓存基线身份/ID绑定、一次archiveSHA/compile与独立子进程PID/CWD/CUDA/GPU/log增长核实。原source数据与权限验证沿用，不新增签名或重验流程。失败保留，不自动重试；默认DURA及原训练测试保持。
