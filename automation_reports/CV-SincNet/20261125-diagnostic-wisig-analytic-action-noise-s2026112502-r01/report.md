# WiSig 非 DURA 解析信道与方差估计核验

- run_id：`20261125-diagnostic-wisig-analytic-action-noise-s2026112502-r01`
- group_id：`wisig-analytic-estimator-validation`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

留出 30 个源 V WiSig 包，各种子配对默认与增强、Full 与 Residual、六种 practical。只用独立的前导估计和源 clean/received 配对估计，同包独立评分。

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

## 方法和指标

在已经冻结的 360 包源 V 仿真面板中按种子 2026112502 抽取 30 个物理包，每台源接收机与每个日期各 2 个，共 15 组。默认和增强强度、Full 和 Residual 路线、六景共 720 个配对观测，每个估计器使用相同 IQ。两种解析方法均无 DURA 或其他模型参与。默认和增强模拟与上轮源 V 诊断一致。没有读取目标 IQ、目标真值或识别器。

- 前导法从实收短训练字段估计频偏，正则拟合已知 STF 和可见 LTF 上的三抽头响应及整数时延。噪声方差由相位补偿后互不重叠的匹配 STF 块差分功率除以 2。
- 配对法接收同包原始源 IQ 和受损 IQ，网格搜索时延并正则拟合三抽头。该方法适合源端提供监督参考。留出残差功率作为有效噪声方差，其中可含三抽头模型的失配。

两种方法使用 −40 至 40 点的时延网格。作用拟合与留出误差使用第 96 至 255 点的交错 8 点块。独立评分在两份盲估计写定后读取无噪模拟输出和注入噪声参数。时延指标比较估计最高能量抽头与模拟无噪波形对原始 IQ 的最佳单抽头时延。直接 NMSE 用留出点的预测误差除以无噪波形平均功率。形状 NMSE 额外只在拟合块上用模拟参考拟合一个复增益，以去除整体幅相基准差，留出块不参与该拟合。方差偏差为 10 log10(估计值/模拟注入方差)，并报告 ±3 dB 覆盖。噪声基准在经过接收机和 AGC 后与估计量单位一致。

## 同包误差

N607 的独立 CPU 评分已完成（VERIFIED）。真实面板检查显示两份估计各写定 24×30 条结果，评分标记显示 oracle-last、模型和目标读取均为零。合成的已知时延、已知噪声及评分隔离检查通过；独立 P0/P1 审查未发现阻塞。

| 方法 | 强度 | 路线 | 范围 | 时延 ±1 点 | 中位直接 NMSE | 中位形状 NMSE | 直接 NMSE p90 | 中位方差偏差 | 方差 ±3 dB |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| preamble | default | full | combined | 90.56% | 69.64% | 18.27% | 105.21% | 16.20 dB | 19.44% |
| preamble | default | full | practical_low_urban | 80.00% | 76.52% | 23.76% | 111.40% | 1.77 dB | 60.00% |
| preamble | default | residual | combined | 90.56% | 69.64% | 18.28% | 105.21% | 16.20 dB | 19.44% |
| preamble | default | residual | practical_low_urban | 80.00% | 76.52% | 23.75% | 111.39% | 1.77 dB | 60.00% |
| preamble | strong | full | combined | 84.44% | 76.53% | 20.70% | 118.32% | 10.57 dB | 23.89% |
| preamble | strong | full | practical_low_urban | 60.00% | 108.75% | 64.11% | 159.37% | 0.76 dB | 66.67% |
| preamble | strong | residual | combined | 84.44% | 76.53% | 20.70% | 118.32% | 10.57 dB | 23.89% |
| preamble | strong | residual | practical_low_urban | 60.00% | 108.75% | 64.10% | 159.36% | 0.76 dB | 66.67% |
| paired | default | full | combined | 97.22% | 0.57% | 0.56% | 19.10% | 3.24 dB | 46.11% |
| paired | default | full | practical_low_urban | 93.33% | 7.28% | 6.23% | 43.76% | 1.40 dB | 90.00% |
| paired | default | residual | combined | 97.22% | 0.57% | 0.56% | 19.11% | 3.24 dB | 46.11% |
| paired | default | residual | practical_low_urban | 93.33% | 7.28% | 6.23% | 43.77% | 1.40 dB | 90.00% |
| paired | strong | full | combined | 93.89% | 0.63% | 0.61% | 31.19% | 1.33 dB | 81.67% |
| paired | strong | full | practical_low_urban | 76.67% | 12.73% | 9.75% | 124.21% | 0.88 dB | 100.00% |
| paired | strong | residual | combined | 93.89% | 0.63% | 0.61% | 31.19% | 1.33 dB | 81.67% |
| paired | strong | residual | practical_low_urban | 76.67% | 12.72% | 9.75% | 124.20% | 0.88 dB | 100.00% |

同一留出点上的零延迟复增益对照，六景中位 NMSE 约为 100.53%。配对解析估计在增强六景对典型包给出 0.63% 的中位直接 NMSE 和 +1.33 dB 的方差偏差；增强低强度城区的中位直接 NMSE 仍为 12.73%，p90 超过 124%。这项小试验证的是在固定源模拟面板上用解析估计近似监督信号的误差。移除模拟器监督后的 DURA 识别性能仍需训练和冻结识别测试的匹配对照。在真实接收机侧，协议前导法的结果只使用实收 IQ。两种方差估计都可能包含未拟合的确定性残差。
