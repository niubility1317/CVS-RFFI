# WiSig 短长前导过渡时标恢复工程对照

- run_id：`20261123-phase1-wisig-timing-engineering-source-s2026092701-r01`
- group_id：`wisig-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

评估仅从当前包的 L-STF/L-LTF 过渡估计时移、修正时间坐标并保持观测幅相，量化对冻结设备识别的独立影响。

该前端用 20 点重复结构估计粗 CFO，在内部消除周期相关中的相位斜率。以标准 Wi-Fi 短训练 8 μs 对应的 200 点为名义过渡，搜索前后两个短周期，联合比较重复残差的变化和 16 点 L-LTF 前缀相关。实际时移取最高分位置。超过名义位置 ±5 点时才考虑修正。较稳健的门在 4,096 个隐藏标签的源 clean U 包上固定第 99.9 百分位，测量非名义位置相对名义窗口的评分余量。校准过程只读取 IQ，完全不需要设备类别、识别器或 practical 轨迹。每条输入单独估计，训练与目标范围保持隔离。[WiSig 原论文](https://arxiv.org/html/2112.15363) · [公开 Wi-Fi 符号同步实现](https://github.com/jhshi/openofdm/blob/master/docs/source/sync_long.rst)

对齐后有效重叠直接复制当前包的 IQ，幅相与数值逐点保持，边界分别检验镜像和基于公共 L-LTF 的缺失后缀外推。后者用实际可见前缀估计复比例，仅用于缺失点。16 点模板不足以辨识完整信道。外推不能作为真实捕获的设备细节证据。两个信道路线应用相同的规则、门限及前导模板。以合法源 V、同一 F50、直通和原 R69 做一次预定五读出对照，完整预测落地后独立评分。

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
