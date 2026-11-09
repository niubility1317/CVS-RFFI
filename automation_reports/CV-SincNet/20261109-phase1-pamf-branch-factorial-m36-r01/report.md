# PAMF 三分支结构消融：第一阶段

- run_id：`20261109-phase1-pamf-branch-factorial-m36-r01`
- group_id：`phase1-pamf-three-branch-factorial`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（2026-11-09 发布后核验；实时状态以远端运行记录为准）

## 目的与对照

去除参考响应，比较时域、频域、功放和三视图路径描述。12结构×3种子；9模型同契约复用、27模型新增训练。仅分类损失监督，raw识别，跨接收机与跨日跨接收机独立评分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无新增模型的正式评分。九个已有同契约模型及其冻结预测、评分已核对并登记复用；其余 27 个模型自动训练，完成后自动进行两组七场景原始识别预测及独立评分。保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

实际执行提交 `17ac02d73c97386a160b1b6e80cb43c5a33fdd7d`；发布包传输一致，远端编译及真实源模型无目标数据启动检查通过。发布后核验见 [launch_verification.json](launch_verification.json)。恢复先核对原运行，不重复启动。
