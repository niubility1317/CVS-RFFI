# ADV3B02-BiCAD-XR设计追踪表

来源：用户提供的《ADV3B02架构级域泛化重构》报告

规格：`docs/superpowers/specs/2026-08-30-adv3b02-bicad-xr-design.md`

状态：`LOCAL_VERIFIED`。实现、协议负测、真实ADV3B02 checkpoint技术兼容smoke和24行矩阵干跑已完成；N607正式性能矩阵尚未完成，因此本文不包含性能结论。

| ID | 要求 | 实现证据 | 状态 | 验证证据与边界 |
|---|---|---|---|---|
| P01 | 保留ADV3B02双骨干、共享Sinc/HF、RCN、CosFace和160维`z_id/z_dom` | `model_dual_cvsincnet.py`、`trainer.py` | verified | `test_model_integration.py`；未替换成HCF-DG单骨干 |
| P02 | 使用`concat_sat_ce_only+LEO_WEAK` | `config.py`、`train_ssdg.py`、launcher | verified | `test_protocol.py`、`test_ssdg_entry.py`；`lambda_sat_cls=0.68`、`lambda_sat_cons=0`、E80开始 |
| P03 | receiver/day/channel因素化头 | `heads.py`、`trainer.py` | verified | `test_heads.py`、`test_trainer.py` |
| P04 | 真实TX one-hot的条件CDAN | `heads.py`、`losses.py` | verified | `test_heads.py`、`test_losses.py`；无标签路径不使用预测标签替代 |
| P05 | 普通batch尽量TX×receiver平衡 | `sampler.py` | verified | `test_sampler.py`；缺cell用mask，不复制样本 |
| P06 | `z_dom`增加独立GRL的TX adversary | `heads.py`、`trainer.py` | verified | `test_heads.py`、`test_trainer.py`；仅`L_s`有TX监督 |
| P07 | V1保持`z_id/z_dom`二分解 | `config.py`、checkpoint runtime | verified | `test_config.py`、`test_ssdg_entry.py`；`z_e+z_int`未实现且未伪报 |
| P08 | TX条件cross-cov替代普通正交 | `losses.py` | verified | `test_losses.py`；`lambda_orth=0` |
| P09 | 每receiver动态ridge donor分类器 | `xdc.py` | verified | `test_xdc.py`；稳定线性求解、停止梯度、不显式求逆 |
| P10 | donor质量加权并蒸馏到公共CosFace头 | `xdc.py`、`trainer.py` | verified | `test_xdc.py`、`test_trainer.py` |
| P11 | XDC每4步执行且向量化 | `config.py`、`trainer.py`、`xdc.py` | verified | `test_xdc.py`、`test_trainer.py`；`xdc_interval=4` |
| P12 | E3小规模clean/satellite配对不变性 | `losses.py`、`trainer.py`、`train_ssdg.py` | verified | `test_ssdg_entry.py`；复用同一concat前向 |
| P13 | 同packet跨receiver配对只在可靠packet ID存在时启用 | 未实现 | deferred | 当前没有可靠packet ID，不恢复26D内容键 |
| P14 | 类条件receiver EMA中心与top-K SVD | `tangent.py`、`trainer.py` | verified | `test_tangent.py`；默认K=4 |
| P15 | F1 factual shift和F2最坏方向shift | `tangent.py`、`trainer.py` | verified | `test_tangent.py`；Stage3后启用 |
| P16 | 三层组EMA风险与0.6/0.3/0.1 CVaR margin-tail | `losses.py`、`metrics.py` | verified | `test_losses.py`、`test_metrics.py` |
| P17 | 两类对抗梯度比监控 | `gradients.py`、`metrics.py` | module_verified | 控制器数值、EMA和artifact schema已测；首轮V1不启用自适应梯度重标定，避免改变冻结损失权重 |
| P18 | shared-stem域正向梯度防火墙=0.05 | `gradients.py`、`trainer.py` | verified | `test_gradients.py`、`test_trainer.py`；仅共享Sinc/HF缩放 |
| P19 | D6每4步局部任务保护投影 | `gradients.py`、`trainer.py` | verified | `test_gradients.py`、`test_trainer.py`；只作用登记模块 |
| P20 | 常驻、稀疏、后期三类总损失路由 | `config.py`、`trainer.py` | verified | `test_config.py`、`test_trainer.py`；不是每步全开的loss soup |
| P21 | 75%普通batch、batch_size96、TX/RX平衡 | `config.py`、`sampler.py` | verified | `test_sampler.py` |
| P22 | 25%结构化`6×4×2`batch | `sampler.py`、`trainer.py` | verified | `test_sampler.py`、`test_trainer.py`；U_s隔离、无placeholder |
| P23 | E3每4步抽8–12对 | `config.py`、`trainer.py` | verified | `test_trainer.py`；V1关闭pair |
| P24 | Stage0–4调度与末段GRL/LR衰减 | `config.py`、`trainer.py` | verified | `test_config.py`、`test_trainer.py`；5000 updates边界已测 |
| P25 | 默认关闭旧机制 | `config.py`、launcher | verified | `test_config.py`、`test_protocol.py`；FastTrust/pseudo/CSD/HCF/LODO/HDRO/open loss/Fishr/MixUp/MixStyle均fail closed |
| P26 | D0–D6、E0–E4、F0–F3候选矩阵 | `config.py`、launcher | verified | `test_config.py`、`test_launcher.py` |
| P27 | 冻结`ADV3B02-BiCAD-XDC-V1` | `config.py` | verified | `test_config.py`；D5+E1+margin-tail |
| P28 | fold1/fold8×3 seeds×5000 updates快筛 | launcher | verified | 24行干跑；8张GPU每卡3行，无重复组合 |
| P29 | 类条件receiver probe artifact | `metrics.py`、`trainer.py` | verified | `test_metrics.py` |
| P30 | `z_dom` TX probe与环境分类证据 | `metrics.py`、`trainer.py` | verified | `test_metrics.py` |
| P31 | 完整donor→query迁移矩阵 | `metrics.py`、`xdc.py` | verified | `test_metrics.py`、`test_xdc.py`；维度与mask已测 |
| P32 | pair干预表示、预测和margin变化 | `metrics.py` | verified | `test_metrics.py`；E3以外显式N/A |
| P33 | Q0.1 margin和最差组合组 | `metrics.py` | verified | `test_metrics.py`；TX/RX/day/channel明确 |
| P34 | strict checkpoint重建后四场景评估 | launcher、`smoke_phase1_bicad_xr_real_checkpoint.py` | verified | 真实历史checkpoint技术smoke通过；正式BiCAD row仍须自身strict runtime与四场景artifact闭合 |
| P35 | 训练辅助模块不进入部署推理图 | model、checkpoint runtime | verified | `test_model_integration.py`；`return_aux=False`保持`z_id→TX`快路径 |
| P36 | Phase1不访问目标receiver、Phase2、support、query或truth | trainer、launcher、smoke | verified | `test_protocol.py`、`test_launcher.py`、真实smoke JSON；访问字段全为false |
| P37 | 记录吞吐、显存、GPU-hours和额外前向比例 | `metrics.py` | verified | `test_metrics.py`；正式报告必须填写实测值 |
| P38 | F3 source-LORO低风险窗口SWAD | `trainer.py`、checkpoint runtime | verified | `test_trainer.py`、`test_ssdg_entry.py`；只在F3启用 |
| P39 | 非连续原始receiver/day编号在域头前映射为连续本地标签 | `train_ssdg.py` | verified | fold1的`3/4/6/8`、fold8的`1/3/4/6`和day1/2/3均经D5真实`compute_step`测试；越界编号fail closed；Task9定点复审`CLEAN` |

## 本地验证汇总

- `phase1_bicad_xr`全套241项通过，仅3条既有AMP弃用警告。
- 相邻`phase1_hcfdg`与ADV3B03回归159项通过。
- 真实`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`技术smoke：严格重建195个状态张量，`missing_keys=[]`、`unexpected_keys=[]`、`shape_mismatches=[]`；fresh BiCAD训练器完成一次optimizer step；clean和3个LEO_WEAK场景前向均为finite。
- 24行干跑：候选`D0/D5/E1/ADV3B02-BiCAD-XDC-V1`，fold1/fold8，seed392001/392002/392003，5000 updates；GPU0–7各3行。
- 独立P0/P1审查发现并修复原始域编号越界风险；仅针对该原问题的定点复审结果为`CLEAN`。

## 当前剩余边界

1.P13继续`deferred`，不得为凑完整度伪造packet ID。
2.历史ADV3B02 checkpoint不含`bicad_xr_runtime`；真实smoke只证明历史基座兼容fresh BiCAD训练与LEO_WEAK前向，不冒充完整BiCAD checkpoint恢复。
3.N607正式矩阵完成前没有性能结论；每行必须以自身final checkpoint完成strict恢复、clean和三种LEO_WEAK评估，才允许写`ARTIFACTS_COMPLETE`。
4.正式发布前必须再次盘点GPU2上的无关进程；本run在该卡的第三行应排队或等待容量释放，不能影响无关任务。
5.r1启动发现from-scratch路径未解析`sample_rate_hz=0`，24行在模型构造前一致技术失败，无性能结果；r2已在入口补WiSig 25MHz默认值并在正式命令显式传值，且用真实模型构造回归复现验证。
