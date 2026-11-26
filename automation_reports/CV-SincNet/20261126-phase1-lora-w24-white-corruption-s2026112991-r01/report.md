# LoRa W24 等中心能量白噪声训练与原混合扰动对照（Full）

- run_id：`20261126-phase1-lora-w24-white-corruption-s2026112991-r01`
- group_id：`phase1-lora-w24-restoration-performance`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（Full 源域训练）

## 目的与对照

Compare archived mixed corruption and stationary Gaussian source observations at identical W24 center-window error energy.

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


## 预登记对照

检验训练扰动形态是否是阻碍 Full 同包收益的因素。两组严格加载同一份合法 40,000 步源 U 首尾 DURA，重置优化器，保留原末期联合损失和相同 1,800 步预算、批量 16、无额外校准。两组对同一个源 U 包的两份独立通用扰动，消耗相同的源 RNG，并同时抽取独立复高斯样本。对照直接采用原混合受扰观测，另一路对 1 MS/s 白噪声应用原生 W24 恢复器前端，在每包每观测的八个中心窗中定标，使新增扰动相对 clean 的能量与对照相同。白噪声的相关性由相同接收前端产生。保留原有 clean 恒等和自然弱差异探针。固定的 500 个 U 留出包两组使用同一归档通用扰动程序。训练不读 practical、仿真轨迹或识别器。

推理沿用双阶段四次 CG、中心固定半量、原生边缘保持。冻结后同 E104 在同一 300 源 V 包评估 Full 七景。候选六景须超过直通、匹配混合扰动半量、40,000 步原通用半量和首尾父版半量，低城区至少持平四组，clean 至少持平直通。独立评分后若晋级，自动完成 Full 正式三分片和本机独立 truth-last。全量回写只作源诊断。

## 发布核验

执行提交为 `59b0e6567ce34ad6a774c37de629401fb5846038`。19 项定向测试和独立 P0/P1 审查通过。实际归档接口确认两组中心窗能量匹配，下一批随机索引一致。归档比对 VERIFIED，N607 编译通过。控制器 862908 和训练 PID 863660、863661 的 CWD 和命令匹配登记，对应物理 GPU 0、3，各占用 3,898 MiB。两个真实父权重无查询检查通过，均有 1,271,038 参数，W24、TX、V、目标、practical 和轨迹读取均为 0，校准数为 0。训练日志均增长至第 200 步。本机收集评分进程已启动，等待源域晋级及正式预测冻结。
