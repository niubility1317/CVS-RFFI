# PAMF第三阶段：RX24独立无标签流与未来查询集

- run_id：`20261029-phase3-full-pool-rx24-r01`
- group_id：`pamf-phase3-full-data`；类别：`data`；阶段：`Phase3-independent-unlabeled-discovery-and-future-query`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在已全量验证Full eq上创建RX24-18 D1无标签768条与D3未来query1920条，单物理记录单practical residual观测；不加载模型或预测评分。

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

## 本地验证与冻结

实际code/config commit：`e4ebaa2682d8c2e87182a8b8ce36ae5ea5eec4d9`。11项针对性测试PASS，含真实ResidualChannel的三景合成IQ；独立P0/P1审查PASS，审查者同11项通过。仅数据builder，无模型checkpoint、目标预测或GPU，不需要模型smoke/N607发布。RX24-18不在四compact的有界header读回已持久化compact_receiver_headers.json；Full原始验证复用，真实新池尚未构建。

## 真实构建完成与独立读回

VERIFIED/VALIDATED_ONCE。原生PID3368实际命令和增长日志已读回；2688条单观测构建完成，进程终止、complete标记及两个public目录独立读取通过。U768（每景256）/Q1920（每景640），canonical接收记录全局唯一，跨role/scene互斥，只有2次ZIP/PKL读取。32行完整日志无错误。未加载模型/预测/query拟合，零GPU，未停止其他owner。实际code/config commit：`e4ebaa2682d8c2e87182a8b8ce36ae5ea5eec4d9`。原始Full验证继续复用；builder完成不代表拒识、适应或注册指标已达标。

两个数据句柄和复核路径见docs/PAMF_PHASE3_FULL_POOL_20261029.md及本地analysis_summary.json、build_post_state.json；私有truth保持独立，原始数据/输出不提交Git。
