# PAMF第三阶段：完整基座源端辅助开放数据

- run_id：`20261029-phase3-source-oe-data-r01`
- group_id：`pamf-phase3-source-open-data`；类别：`data`；阶段：`Phase3-source-auxiliary-open-development-data`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

按元数据集合/容量固定Full中全compact外的7辅助TX，原source5RX×D1D2D3每cell70，训练暴露4200/新设备验证3150；纯数据，不读正式target、不训模型。

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

## 构建前版本与证据

实际code/config commit：`795e4a9604f391cb53e19ba24c2d2e8d5f24188a`。6项针对性测试及10个subtests通过，独立P0/P1审查PASS且独立6项通过。科学source辅助开放边界已在PROJECT_PROTOCOL定义；这是额外源成员监督，非零源标签，目标仍零标签。纯CPU数据builder，无模型/原L/U/V/正式target访问，不需要checkpoint smoke或N607发布。Full原始验证复用，尚未构建真实7350条记录。

## 真实构建与独立读回

VERIFIED/VALIDATED_ONCE。15个source成员读取完成，7350个canonical物理记录唯一，4TX4200辅助训练与3TX3150独立源验证互斥；每TX1050、每TX/RX/day70。两个public各3文件，Main source域映射15组，独立有限值/形状/ID/域覆盖读回通过；原生PID55544实际命令曾核实，完成后退出。17行完整日志无异常。原L/U/V、模型、正式target和channel生成均0，零GPU，不改变其他owner。实际code/config commit：`795e4a9604f391cb53e19ba24c2d2e8d5f24188a`。这只是数据入口，额外源成员监督已披露，不证明95%或注册成功。
