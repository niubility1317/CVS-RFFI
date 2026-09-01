# ADV3B02物理因子化交叉重构设计追踪表

设计来源：`E:\codex\home\attachments\1d377d1c-fe81-4e09-ac71-dc857e445413\pasted-text.txt`

批准路线：并行`ADV3B02-FCR`分支，采用方案A相对链路语义；用户于2026-09-01确认聊天设计。

实现规格：`docs/superpowers/specs/2026-09-01-adv3b02-factorized-cross-reconstruction-design.md`

实施计划：`docs/superpowers/plans/2026-09-01-adv3b02-factorized-cross-reconstruction.md`

| ID | 来源章节 | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| FCR-01 | 总体判断、第二节 | clean/LEO必须是同一物理片段、同一内容和同一TX的干预式配对 | `code/baseline_origin_sat_view.py`、`code/cvsrffi/phase1_fcr_interventions.py` | verified | `test_phase1_fcr_pairing.py`核对physical ID、crop和view | 不产生第二个Phase2观测 |
| FCR-02 | 第一、十三节 | Decoder遵循内容生成→TX响应→链路/接收机的物理顺序 | `code/cvsrffi/phase1_fcr_decoder.py` | implemented | 本地模块测试核对内容→指纹→信道/接收机调用顺序、形状和全部latent梯度 | 禁止普通latent concat Decoder；端到端和N607证据待后续任务 |
| FCR-03 | 第六节6.1 | `z_s`是低采样率时序token并承担内容/激励，不携带TX/receiver/domain | `code/cvsrffi/phase1_fcr_factors.py` | implemented | 本地聚焦测试核对`[B,64,32]`时序token、masked reconstruction梯度和默认detach身份输入；独立probe待后续任务 | TX CE默认不更新`E_s`；未形成端到端或N607证据 |
| FCR-04 | 第六节6.2 | `z_f=[z_f_id,z_tx_state]`，作为激励条件化响应算子参数 | `code/cvsrffi/phase1_fcr_factors.py`、`code/cvsrffi/phase1_fcr_fingerprint.py` | implemented | 本地模块测试核对`z_f_id:[B,160]`单位范数和`z_tx_state:[B,16]`输出 | `L_id`只作用于`z_f_id`；跨天稳定性和端到端证据待后续任务 |
| FCR-05 | 第六节6.2 | `G_f(e,z_f)`由物理基和受限小残差产生`delta_f` | `code/cvsrffi/phase1_fcr_fingerprint.py` | implemented | 本地模块测试核对固定基、公共相位等变、零输入有限和逐样本能量上限 | 受限残差只读excitation和`z_tx_state`；未形成端到端或N607证据 |
| FCR-06 | 第六节6.3 | `z_n=[z_ch,z_rx,z_sync,z_gain]`且为低容量结构化latent | `code/cvsrffi/phase1_fcr_nuisance.py` | implemented | 本地模块测试核对33维容量、四段形状、无skip/时序latent和零/近零输入有限有界 | 禁止目标波形旁路；类别泄漏probe待后续任务 |
| FCR-07 | 第七节 | 噪声以条件均值和方差建模，不精确重构noise realization | `code/cvsrffi/phase1_fcr_decoder.py`、`code/cvsrffi/phase1_fcr_losses.py` | implemented | 本地Decoder测试核对条件均值、方差上下界、零内容有限和latent梯度；异方差NLL待Task6 | `sigma`不能无限增大；未形成端到端或N607证据 |
| FCR-08 | 第八节8.1-8.3 | 重构误差包含受限对齐、MRSTFT和幅度门控共轭相位增量 | `code/cvsrffi/phase1_fcr_losses.py` | implemented | 本地聚焦测试核对有界异方差NLL排序、三窗MRSTFT噪声地板、phase wrap、零输入有限及预测梯度 | 不直接对wrapped phase做L1；未形成端到端或N607证据 |
| FCR-09 | 第八节8.4 | `R_fp`是冻结物理特征集合并受Fisher可辨识性门控 | `code/cvsrffi/phase1_fcr_physics.py` | implemented | 本地聚焦测试核对零参数确定性八块特征、Gram有效秩/覆盖/PAPR/SNR/噪声地板detach质量和低PAPR下调PA | 不与Decoder自由协同训练；未形成端到端或N607证据 |
| FCR-10 | 第九节 | 交叉生成结果重新编码并恢复来源`z_s/z_f`及目标`z_n` | `code/cvsrffi/phase1_fcr_losses.py` | pending | 双向latent-cycle合成测试 | 所有参考latent使用stop-gradient |
| FCR-11 | 第十节 | clean/LEO显式共享一致性并配置防塌缩约束 | `code/cvsrffi/phase1_fcr_losses.py` | pending | variance、covariance和常数塌缩负测 | 不最大化`z_n`无界距离 |
| FCR-12 | 第十节 | `z_n`回归或分类已知Doppler、SNR、delay、rate、taps、SFO、STO | `code/cvsrffi/phase1_fcr_nuisance.py`、`code/cvsrffi/phase1_fcr_losses.py` | pending | 已知增强参数恢复测试 | 使用免费模拟监督 |
| FCR-13 | 第十一节 | 构造Nuisance、Content、Fingerprint三轴干预立方体 | `code/cvsrffi/phase1_fcr_interventions.py` | blocked | 合成夹具验证严格索引；真实WiSig能力未测量 | 未发现可只读使用的本地WiSig索引/公共前导配置；禁止把合成Fingerprint Pair写成真实证据 |
| FCR-14 | 第十二节 | 采用方案A，`z_n^leo`解释相对clean的新增复合nuisance | 规格、正式配置和报告 | pending | 配置标记和claim boundary测试 | 不声明纯星地信道恢复 |
| FCR-15 | 第十三节 | Canonicalizer输出`x_tilde/eta_hat/r_can`并保留细粒度TX残差 | `code/cvsrffi/phase1_fcr_canonicalizer.py` | implemented | 本地合成测试核对公共gain/phase/CFO误差下降与非公共IQ imbalance残差能量保持 | 初始实现保守解析归一化；未形成端到端或N607证据 |
| FCR-16 | 第十四节1-6 | 实现`L_id/self/swap/shared/latent-cycle/eta` | `code/cvsrffi/phase1_fcr_losses.py` | pending | 每项独立单测和训练可达性测试 | swap为clean↔LEO双向 |
| FCR-17 | 第十四节7 | 因子泄漏抑制要求`z_f/z_n/z_s`各自高目标信息、低非目标信息 | `code/cvsrffi/phase1_fcr_losses.py`、`code/cvsrffi/phase1_fcr_diagnostics.py` | pending | 条件域混淆、cross-covariance和独立probe | 不完全依赖全局DANN |
| FCR-18 | 第四、五、十四节8 | 改进necessity为定向移植、保持内容/nuisance、同TX和drop-f三角验证 | `code/cvsrffi/phase1_fcr_transplant.py` | pending | 独立冻结分类器和重编码测试 | shuffle gap单独不算通过 |
| FCR-19 | 第十四节9 | 实现指纹能量、响应平滑、参数边界和物理特征约束 | `code/cvsrffi/phase1_fcr_physics.py`、`code/cvsrffi/phase1_fcr_losses.py` | implemented | 本地聚焦测试核对逐物理块Fisher加权、零权重归零、零/近零有限和预测梯度；模块提供指纹能量、响应平滑及参数边界罚项 | 物理项受Fisher gate控制；未形成端到端或N607证据 |
| FCR-20 | 第十五节 | `U_s`只使用无标签自监督项，不能读取隐藏TX真值 | `code/dataset_wisig.py`、`code/cvsrffi/phase1_fcr_interventions.py`、`code/train.py` | implemented | `test_phase1_fcr_pairing.py`和`test_phase1_fcr_interventions.py`验证元数据不可逆TX边界 | 本任务仅完成并验证U_s元数据边界；训练梯度路由由后续任务接线验证 |
| FCR-21 | 第十六节 | 训练按基础重构→swap/cycle→定向移植→身份DG四阶段启用 | `code/cvsrffi/phase1_fcr_schedule.py`、`code/train.py` | pending | E1-40/E41-90/E91-150/E151-200启用矩阵测试 | 权重采用ramp而非瞬时全开 |
| FCR-22 | 第十七节 | 保存latent纯度、配对距离、移植、参数恢复和资源诊断 | `code/cvsrffi/phase1_fcr_diagnostics.py`、`code/train.py` | pending | 日志字段完整性测试 | probe独立于训练分类器 |
| FCR-23 | 第十七节 | 实现R0-R8递进消融并绑定同row结果 | `code/scripts/`、正式实验报告 | pending | launcher dry-run和结果行绑定测试 | 先单seed最小可证伪矩阵 |
| FCR-24 | 项目协议4、4.3节 | 保持`L_s/U_s/V`权限、LEO_WEAK日程和clean/三场景最终评测 | `code/train.py`、正式launcher | pending | 协议负测、真实checkpoint无query smoke、四评测检查 | 不修改普通ADV3B02全局默认 |
| FCR-25 | ADV3B02集成要求 | `use_fcr=false`保持旧checkpoint、state和输出兼容 | `code/model_dual_cvsincnet.py`、`code/cvsrffi/checkpoint.py` | verified | 严格加载和逐元素关闭态测试 | 仅开启时实例化FCR参数 |
| FCR-26 | 部署闭合 | checkpoint保存FCR模块、物理基、统计和feature schema；单LEO IQ独立推理 | `code/cvsrffi/checkpoint.py`、`code/model_dual_cvsincnet.py` | pending | 保存—加载—单视图推理往返 | 推理不需要clean伴随输入 |

## 当前计数

- `verified`：2
- `implemented`：8
- `deferred`：0
- `rejected`：0
- `blocked`：1
- `pending`：17

## 最高风险项

FCR-13的Fingerprint Pair要求同内容、同链路、不同TX。实现前必须确认WiSig公共preamble、窗口定位和receiver/day/激励匹配是否足以构造严格pair。如果证据不足，该项转为`blocked`，不能用随机异TX配对替代后宣称完整设计一致。

## 一致性判定

只有FCR-01至FCR-26全部达到`verified`，或少数条目以明确理由进入`deferred/rejected/blocked`且最终声明同步收缩，才能报告实现状态。任何CLI未接线、模块不可达、诊断未输出或用ECRS局部响应辨识代替完整FCR的版本均不是严格设计一致。
