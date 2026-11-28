# LoRa DURA 对照的数据契约核查

本记录服务于 LoRa 与 WiSig 融合研发，不改变固定的 DURA v1.0。结论依据代码、已冻结模型及历史报告；本次核查未生成新预测，也未读取目标真值。

## 固定版本与 LoRa 历史适配

| 路线 | 权重和输入 | 可复用范围 |
| --- | --- | --- |
| 固定 DURA v1.0 | `20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt`，源数据为 original7/63/30；输入 `[B,2,256]`；固定 Graph-RR 推理路径 | 已有 WiSig 契约内的冻结评估。该权重及其成绩没有自动获得 LoRa 数据契约下的继承资格。 |
| LoRa Full R69 载波坐标适配 | `20261122-phase1-lora-dura-r69-aligned-m2-r01`；R65 从零训练 2,400 步，R69 从同契约教师初始化并训练 2,400 步；六个 256 点 chirp，去载波并归一化；源 L/U 和模拟 clean 配对，不读 F、TX 标签、V 或目标进行更新 | 与 LoRa Full 源物理合同匹配的 R69 架构适配。原预测连接 Tri-set；采用逐包 15% 限幅、半量和源门控。历史成绩不能直接充当 W24 E104 的结果，也不能称为原固定权重。 |
| LoRa residual R69 | `20261123-phase1-lora-dura-residual-practical-m1-r01`；从零的同契约八阶段训练，总计 39,520 步，固定 R69 为末阶段；六个 256 点 chirp | Residual 信道训练和历史对照；快照和噪声注入与 Full 不同。`w24_residual_pilot.py` 包含 W24 接口，但它的 residual 预测不能替代 Full 比较。 |
| LoRa Full `dura_half` | `full_dura.py` 的 `RiskDura` 加 LoRa detail prior 和半量修正 | 可作为其自身历史路线，代码显示其并非固定 R69 `StageDistilled`。 |

可核查入口：`experiments/pamf_stage2_inference/frozen_version.json`、`experiments/pamf_lora_main/full_dura_r69_lora.py`、`experiments/pamf_lora_main/full_dura.py`、`experiments/pamf_lora_main/full_dura_detail.py`、`experiments/pamf_lora_main/w24_residual_pilot.py`。训练与测试来源分别见 `20261122-phase1-lora-dura-r69-aligned-m2-r01` 和 `20261123-phase1-lora-dura-residual-practical-m1-r01` 报告。

## 可实施的匹配对照

对于 Full 路线，可固定 LoRa Full R69 学生及其同源门控，并在同一个 8,192 点接收包上，沿用已登记的六个 256 点窗提取与回写路径，连接冻结 W24 E104 原默认头。该组合的逐包回写及读出属于新适配，需要登记版本、核验输入和来源并完成小规模预测，不能复用旧 Tri-set 结果冒充它的结果。预测应先固定，再独立连接源 V 真值；只与相同物理 ID、同一次信道观测和相同读出的结果配对。预测规则、接收机、样本、度量与停止规则应在读数前固定。

已有目标 Full 七景及部分源 V 已经曝光。任何在这些数据上做出的对照只能按实际曝光背景解释；后续候选正式验收仍优先使用未参与筛查和方案选择的保留数据。

## 与实增益及当前研发的关系

LoRa 原生实增益版本 `20261128-phase1-lora-raw8192-frequency-quadrature-s2026113037-r01` 的冻结 E104 Full 结果为六景均值 82.2967%、低城区 62.1167%、clean 95.5467%，均为 30,000 包每景。此成绩属于其自身 Full 预测和识别器；其完整模型训练来源和许可需随新初始化逐项核验。固定 DURA v1.0、LoRa Full R69 适配和本轮用户优先的 R02/R07/R17 各保留独立身份。

截至本记录，LoRa 适配 R69 接 E104、使用当前筛查相同 Full 接收包的预测与评分仍待完成。WiSig 使用与其契约匹配的固定 R69 的源筛查对照已分别记录在 `20261128-diagnostic-wisig-r69-screen-s2026112868-r01` 和 `20261128-diagnostic-wisig-r69-joint-s2026112872-r01`，这些源 V 诊断不是两个数据集的正式验收结果。
