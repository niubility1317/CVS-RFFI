# LoRa W24 强扰动半量风险重分配（Full）

- run_id：`20261126-phase1-lora-w24-severity-weight-s2026112991-r01`
- group_id：`phase1-lora-w24-restoration-performance`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（Full 源域续训）

## 目的与对照

Test a bounded source-only severity redistribution of the deployed half-readout loss without changing the source observations.

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

检验强受扰源包在末级半量损失中获得更高权重，是否改善 Full 同包识别。旧窄窗尾部目标在每个估计噪声层中关注波形误差上尾。本次以同条合法源 IQ 的实际中心窗扰动能量比衡量难度，在原生 W24 半量目标中平滑调整两份独立配对观测的权重。权重取 1 + 3s/(0.1 + s)，批内归一。首尾两条和中间六条各占配对损失的一半。两组共享原通用扰动、所有配对波形及其随机流。只调整受扰两份半量观测对损失的贡献。原合成弱差异探针、clean 恒等、其他阶段和物理一致性损失沿用。恢复器部署沿用只读接收 IQ 的双阶段四次 CG 和固定半量回写。

两臂严格加载同一个合法源 U 训练并按 40,000 步冻结的首尾权重，重置 AdamW，新增 1,800 步，固定批量 16，无校准。训练只读取合法源 U IQ。共同 500 个留出 U 包保持原扰动和无权目标，另用每批相对扰动能量最高的四分之一计算诊断风险。W24、源 V、TX、目标或仿真轨迹均不参与拟合。

冻结后由 E104 对同一 300 个源 V 包做 Full 七景独立评分。候选半量须在六景超过直通、匹配均匀对照、40,000 步原通用半量和首尾父版，低城区至少持平四组，clean 至少持平直通。父版盲预测须逐包再现。通过后自动跑 Full 的三个正式分片，完整预测固定后独立评分。全量回写仅用于源域诊断。

## 发布核验

执行提交为 `dd24ab10d6059246c92687eaf1a8314e1edb9f80`。20 项定向测试和独立 P0/P1 审查通过，实际训练归档接口确认权重只作用于两份受扰观测，clean、探针和共同留出损失维持原规则。发布归档比对 VERIFIED，远端编译通过。控制器 885604 和训练 PID 885888、885887 的完整命令与 CWD 均独立读回。对照与加权组分别使用物理 GPU 6、3，各占用 3,896 MiB。两组真实父权重无查询检查通过，训练读取 W24、TX、源 V、目标、practical 及轨迹计数均为 0。每组 1,271,038 个参数，日志分别增长至 50、100 步。本机独立评分程序已等待晋级和正式预测冻结。
