# ADV3B02物理因子化交叉重构设计追踪表

设计来源：`E:\codex\home\attachments\1d377d1c-fe81-4e09-ac71-dc857e445413\pasted-text.txt`

批准路线：并行`ADV3B02-FCR`分支，采用方案A相对链路语义；用户于2026-09-01确认聊天设计。

实现规格：`docs/superpowers/specs/2026-09-01-adv3b02-factorized-cross-reconstruction-design.md`

实施计划：`docs/superpowers/plans/2026-09-01-adv3b02-factorized-cross-reconstruction.md`

| ID | 来源章节 | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| FCR-01 | 总体判断、第二节 | clean/LEO必须是同一物理片段、同一内容和同一TX的干预式配对 | `code/baseline_origin_sat_view.py`、`code/cvsrffi/phase1_fcr_interventions.py` | verified | `test_phase1_fcr_pairing.py`核对physical ID、crop和view | 不产生第二个Phase2观测 |
| FCR-02 | 第一、十三节 | Decoder遵循内容生成→TX响应→链路/接收机的物理顺序 | `code/cvsrffi/phase1_fcr_decoder.py`、`code/model_dual_cvsincnet.py` | implemented | 模块顺序测试及`test_phase1_fcr_forward.py`核对单视图聚合可达、`fcr_decode`形状、有限值和反传 | 禁止普通latent concat Decoder；已具本地端到端可达性，训练、真实checkpoint和N607证据待后续任务 |
| FCR-03 | 第六节6.1 | `z_s`是低采样率时序token并承担内容/激励，不携带TX/receiver/domain | `code/cvsrffi/phase1_fcr_factors.py`、`code/model_dual_cvsincnet.py` | implemented | 模块测试及`test_phase1_fcr_forward.py`核对默认`[B,64,32]`语义、随`input_len`缩放的输出形状、有限值和内容路径反传 | TX CE默认不更新`E_s`；已具本地端到端可达性，独立probe、训练和N607证据待后续任务 |
| FCR-04 | 第六节6.2 | `z_f=[z_f_id,z_tx_state]`，作为激励条件化响应算子参数 | `code/cvsrffi/phase1_fcr_factors.py`、`code/cvsrffi/phase1_fcr_fingerprint.py`、`code/model_dual_cvsincnet.py` | implemented | 模块测试及`test_phase1_fcr_forward.py`核对`z_f_id:[B,160]`单位范数、`z_tx_state:[B,16]`、`z_id_raw is z_id`和身份路径反传 | `L_id`只作用于`z_f_id`；已具本地端到端可达性，跨天稳定性、训练和N607证据待后续任务 |
| FCR-05 | 第六节6.2 | `G_f(e,z_f)`由物理基和受限小残差产生`delta_f` | `code/cvsrffi/phase1_fcr_fingerprint.py`、`code/model_dual_cvsincnet.py` | implemented | 模块测试及`test_phase1_fcr_forward.py`核对聚合调用、complex64`delta_f`、有限质量诊断和FCR参数反传 | 受限残差只读excitation和`z_tx_state`；聚合边界detach响应激励以避开Task4已提交原地基的反传版本冲突，内容仍经Decoder路径反传；训练和N607证据待后续任务 |
| FCR-06 | 第六节6.3 | `z_n=[z_ch,z_rx,z_sync,z_gain]`且为低容量结构化latent | `code/cvsrffi/phase1_fcr_nuisance.py`、`code/model_dual_cvsincnet.py` | implemented | 模块测试及`test_phase1_fcr_forward.py`核对顶层`channel/receiver/sync/gain`精确结构、形状、有限值和Decoder反传 | 禁止目标波形旁路；已具本地端到端可达性，类别泄漏probe、训练和N607证据待后续任务 |
| FCR-07 | 第七节 | 噪声以条件均值和方差建模，不精确重构noise realization | `code/cvsrffi/phase1_fcr_decoder.py`、`code/cvsrffi/phase1_fcr_losses.py`、`code/model_dual_cvsincnet.py` | implemented | Decoder/NLL模块测试及`test_phase1_fcr_forward.py`核对条件均值、有限有界方差输出和聚合反传 | `sigma`不能无限增大；已具本地端到端可达性，训练、真实checkpoint和N607证据待后续任务 |
| FCR-08 | 第八节8.1-8.3 | 重构误差包含受限对齐、MRSTFT和幅度门控共轭相位增量 | `code/cvsrffi/phase1_fcr_losses.py` | implemented | 本地聚焦测试核对有界异方差NLL排序、三窗MRSTFT噪声地板、phase wrap、零输入有限及预测梯度 | 不直接对wrapped phase做L1；未形成端到端或N607证据 |
| FCR-09 | 第八节8.4 | `R_fp`是冻结物理特征集合并受Fisher可辨识性门控 | `code/cvsrffi/phase1_fcr_physics.py` | implemented | 本地聚焦测试核对零参数确定性八块特征、Gram有效秩/覆盖/PAPR/SNR/噪声地板detach质量和低PAPR下调PA | 不与Decoder自由协同训练；未形成端到端或N607证据 |
| FCR-10 | 第九节 | 交叉生成结果重新编码并恢复来源`z_s/z_f`及目标`z_n` | `code/cvsrffi/phase1_fcr_losses.py` | implemented | `test_phase1_fcr_cross_losses.py`核对双向回编码调用、目标detach和无效pair归零 | 所有参考latent使用stop-gradient；未形成端到端或N607证据 |
| FCR-11 | 第十节 | clean/LEO显式共享一致性并配置防塌缩约束 | `code/cvsrffi/phase1_fcr_losses.py` | implemented | `test_phase1_fcr_cross_losses.py`核对双向stop-gradient、variance/covariance和常数塌缩正罚项 | 不最大化`z_n`无界距离；未形成端到端或N607证据 |
| FCR-12 | 第十节 | `z_n`回归或分类已知Doppler、SNR、delay、rate、taps、SFO、STO | `code/cvsrffi/phase1_fcr_nuisance.py`、`code/cvsrffi/phase1_fcr_losses.py` | implemented | `test_phase1_fcr_cross_losses.py`核对`nuisance_valid`字段掩码和无效字段零梯度 | 使用免费模拟监督；未形成端到端或N607证据 |
| FCR-13 | 第十一节 | 构造Nuisance、Content、Fingerprint三轴干预立方体 | `code/cvsrffi/phase1_fcr_interventions.py` | blocked | 合成夹具验证严格索引；真实WiSig能力未测量 | 未发现可只读使用的本地WiSig索引/公共前导配置；禁止把合成Fingerprint Pair写成真实证据 |
| FCR-14 | 第十二节 | 采用方案A，`z_n^leo`解释相对clean的新增复合nuisance | `code/train.py`、Task10报告 | implemented | 候选路径用同物理clean/LEO pair将LEO factor作为cross decode目标nuisance并记录相对链路claim boundary | 仅为成对训练语义；不声明纯星地信道恢复或真实实验效果 |
| FCR-15 | 第十三节 | Canonicalizer输出`x_tilde/eta_hat/r_can`并保留细粒度TX残差 | `code/cvsrffi/phase1_fcr_canonicalizer.py` | implemented | 本地合成测试核对公共gain/phase/CFO误差下降与非公共IQ imbalance残差能量保持 | 初始实现保守解析归一化；未形成端到端或N607证据 |
| FCR-16 | 第十四节1-6 | 实现`L_id/self/swap/shared/latent-cycle/eta` | `code/cvsrffi/phase1_fcr_losses.py`、`code/train.py` | implemented | 既有cross-loss测试加`test_phase1_fcr_gradient_routing.py`核对完整Task6/7/8组合、双向swap/cycle和有限反传；定点回归核对正式identity CE只消费`fcr_tx_logits(z_f_id)`和显式`L_s`label mask | 已形成候选训练本地可达性；未形成真实训练结果或N607证据 |
| FCR-17 | 第十四节7 | 因子泄漏抑制要求`z_f/z_n/z_s`各自高目标信息、低非目标信息 | `code/cvsrffi/phase1_fcr_losses.py`、`code/cvsrffi/phase1_fcr_diagnostics.py` | implemented | `test_phase1_fcr_cross_losses.py`核对cross-covariance组合和训练外probe接口 | 条件domain-confusion由调用方显式提供；不完全依赖全局DANN |
| FCR-18 | 第四、五、十四节8 | 改进necessity为定向移植、保持内容/nuisance、同TX和drop-f三角验证 | `code/cvsrffi/phase1_fcr_transplant.py` | implemented | 本地定向移植测试核对严格索引、空pair零调用、冻结分类器输入梯度、内容/nuisance保持、同TX控制、drop-f stop-gradient和Decoder冻结 | FCR-13仍blocked；本条仅为模块语义实现，不构成真实strict-pair、端到端或N607证据；shuffle gap单独不算通过 |
| FCR-19 | 第十四节9 | 实现指纹能量、响应平滑、参数边界和物理特征约束 | `code/cvsrffi/phase1_fcr_physics.py`、`code/cvsrffi/phase1_fcr_losses.py` | implemented | 本地聚焦测试核对逐物理块Fisher加权、零权重归零、零/近零有限和预测梯度；模块提供指纹能量、响应平滑及参数边界罚项 | 物理项受Fisher gate控制；未形成端到端或N607证据 |
| FCR-20 | 第十五节 | `U_s`只使用无标签自监督项，不能读取隐藏TX真值 | `code/dataset_wisig.py`、`code/cvsrffi/phase1_fcr_interventions.py`、`code/train.py` | verified | `test_phase1_fcr_unlabeled_boundary.py`、gradient测试和定点回归核对`labels=-1`、显式mask、正式0.07/0.63/0.30比例fail-closed及`U_s`identity/prototype精确零 | 验证范围仅限本地U_s训练路由；无真TX、硬伪标签或query证据面 |
| FCR-21 | 第十六节 | 训练按基础重构→swap/cycle→定向移植→身份DG四阶段启用 | `code/cvsrffi/phase1_fcr_schedule.py`、`code/cvsrffi/schedule.py`、`code/train.py` | implemented | `test_phase1_fcr_schedule.py`核对E1/40/41/90/91/150/151/200、线性ramp、optimizer step交替和越界拒绝 | FCR权重独立于既有E80 sat CE/LEO_WEAK；未形成真实checkpoint证据 |
| FCR-22 | 第十七节 | 保存latent纯度、配对距离、移植、参数恢复和资源诊断 | `code/cvsrffi/phase1_fcr_diagnostics.py`、`code/train.py` | verified | `test_phase1_fcr_diagnostics.py`验证17键、独立detach probe、训练外collector和严格pair缺失时`N/A`+原因；完整聚焦组79 passed | probe独立于训练分类器；当前真实smoke不具备严格移植pair，正式row按能力报告`N/A`而非0 |
| FCR-23 | 第十七节 | 实现R0-R8递进消融并绑定同row结果 | `code/scripts/launch_phase1_adv3b02_fcr_20260901.sh`、`code/train.py`、正式实验报告 | verified | `test_phase1_adv3b02_fcr_launcher.py`验证R0-R8显式递进、14类checkpoint/log/diagnostics/prediction路径全部位于row root且跨row两两隔离、E200/E80/三段LEO与四评测配置；Python CLI dry-run通过 | 只验证正式入口和同row artifact绑定，尚未生成任何真实消融结果；先单seed最小可证伪矩阵 |
| FCR-24 | 项目协议4、4.3节 | 保持`L_s/U_s/V`权限、LEO_WEAK日程和clean/三场景最终评测 | `code/train.py`、`code/cvsrffi/phase1_fcr_schedule.py`、正式launcher | pending | 协议负测核对L_s/U_s/V、query不可达、只读V；真实source/checkpoint无query smoke已闭合clean/三LEO逐样本prediction并验证ID、row/run/schema/route完整性 | 正式launcher只写`PREDICTIONS_READY`，必须在独立truth-last scorer后才能进入`ARTIFACTS_COMPLETE`；尚无四场景最终数值评测 |
| FCR-25 | ADV3B02集成要求 | `use_fcr=false`保持旧checkpoint、state和输出兼容 | `code/model_dual_cvsincnet.py`、`code/cvsrffi/checkpoint.py` | verified | 严格加载、逐元素关闭态和普通eval/Meta-SSL回归测试通过 | 仅开启时实例化FCR参数及`fcr_identity_head`；普通路径继续消费legacy输出 |
| FCR-26 | 部署闭合 | checkpoint保存FCR模块、物理基、统计和feature schema；单LEO IQ独立推理 | `code/cvsrffi/checkpoint.py`、`code/model_dual_cvsincnet.py` | verified | checkpoint单测及真实ADV3B02 epoch194 checkpoint+合法Phase1 source样本smoke完成FCR backward、bundle严格恢复、单`leo_clear_weak`的`z_f_id`/`fcr_tx_logits`精确复现和四场景prediction导出 | bundle显式记录`fcr_identity_head(z_f_id)`；旧关闭态无bundle兼容；推理不需要clean伴随输入 |

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
