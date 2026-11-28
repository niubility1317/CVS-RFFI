# LoRa 与 WiSig 共用冻结识别器置信度辅助专家路由筛查

- run_id：`20261128-phase1-shared-confidence-router-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：PARTIAL；WiSig 源筛查已评分且未晋级，LoRa 首步技术失败后在独立 run 补跑。

## 目的与对照

共享物理特征和冻结识别器类别无关置信度，基于源伪标签训练当前包专家路由。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 d799c59a13fcc9ff2972abbe1e111ff0612870ee；控制器 3918976，WiSig worker 3919052 / GPU 1 首步通过并继续。LoRa worker 3919051 在训练前报 NamespacePath 无 insert，失败日志保留，独立 LoRa 技术补跑 20261128-phase1-shared-confidence-lora-retry-s2026112957-r01。预检、归档 SHA、远端编译及独立 P0/P1 审查通过；先前 WiSig 导入 P1 已修复并定向复核，LoRa 新故障另行定向修复。

## 结果与覆盖

同一物理特征加识别器置信度网络与源伪标签损失；LoRa 实增益、WiSig R17 是已曝光源 V 选定的锚点，各自独立权重。训练及推理均对逐包归一化的两个专家输出调用本数据集冻结 F；推理每包比单次最终分类多两次 F 前向，必须另行披露时延和资源。LoRa W24 E104 使用原 scipy 前向、WiSig 使用 Tri-SIG-TIME SSL F0；不读源 U 真实 TX、V 或目标训练。源 V 经多轮研发曝光；WiSig 筛查完成，LoRa 筛查尚未完整。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### WiSig 固定源 V 筛查

WiSig 低城区直通/R17/实增益/DURA v1/置信度路由分别为 200/332/190/313/332，clean 均为 360/360。相对 R17 改对 0、改错 0，净 0，固定半区均为 0，95% 配对区间 [0,0]；未达到晋级条件。正确包集合相同并不证明所有错误预测类别相同。逐对照见本地 `readback/wisig/screen-score.json`。源 V 多轮曝光，未读取目标数据。

路由网络 11,114 参数，源 U 曝光 24,000 次，教师样本前向 120,048 次，含首步检查；有效伪标签 16,804 次，其中纠错目标 3,523 次。训练 283.07 秒，PyTorch 峰值约 58.9 MiB；两专家和冻结识别器的祖先成本、每包额外两次 F 推理及实际时延另计。LoRa 结果以独立技术补跑为准。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
