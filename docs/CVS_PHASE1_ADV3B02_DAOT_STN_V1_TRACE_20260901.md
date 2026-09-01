# ADV3B02-DAOT-STN-V1报告实现追踪

## 适用范围

- 来源：用户提供的《九、第六项关键创新：部署分布匹配的信道轨道教师与选择性切空间正则》。
- 方法ID：`ADV3B02-DAOT-STN-V1`。
- 当前协议：Phase1 source-only，保持`L_s/U_s/V=0.07/0.63/0.30`、V只读、query不可达。
- 基线边界：保留`ADV3B02_FASTTRUST_EFF`的hard集合、class-balanced cap、source prior和`hard∩sat_mask`路径。
- 用户修订：新方法默认使用性能优先的三教师视图`clean+medium+hard`；Temporal Orbit Memory仅作为A8效率对照。
- 声明边界：没有真实LEO参数统计时只称为`deployment-proxy matched`。
- 隔离交付：实现位于工作树`.worktrees/adv3b02-daot-stn-v1`和分支`codex/adv3b02-daot-stn-v1-20260901`，不与并行方法改动混合。
- 执行边界：本轮只完成报告落地、测试和无query smoke，未启动本地或N607性能实验。

## 追踪矩阵

| ID | 来源章节 | 要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| OT-01 | 9.1、9.2 | 以干扰轨道条件分布为对象，不把目标简化成多增强特征相等 | `code/cvsrffi/orbit_teacher.py` | verified | 球面聚合及梯度测试通过 | 特征先归一化 |
| OT-02 | 9.2.2 | 保留clean＋单个LEO学生有梯度结构 | `code/SSDG/train_ssdg.py` | verified | labeled主链集成与反传测试通过 | 原卫星CE的E80起点未修改 |
| OT-03 | 9.3、9.5 | 物理联合代理分布、严重度先验和裁剪后的`p_dep/q_train` | `code/cvsrffi/deployment_orbit.py` | verified | 代理权重、裁剪和真实统计声明负测通过 | 不伪造真实LEO经验分布 |
| OT-04 | 9.4 | 特征可靠性权重与分类置信度权重分离 | `code/cvsrffi/daot_training.py` | verified | feature聚合与logit共识路由测试通过 | feature聚合不读取logit熵 |
| OT-05 | 9.4、9.6 | coverage floor、Huber残差权重、鲁棒球面均值 | `code/cvsrffi/orbit_teacher.py` | verified | coverage、Huber、单位球及`N_eff`测试通过 | `gamma_cov=0.15`、`beta_min=0.30` |
| OT-06 | 9.7 | `L_orb,z/logit/proto/rel`四层目标 | `code/cvsrffi/orbit_teacher.py`、`code/cvsrffi/daot_training.py` | verified | 四类损失行为测试通过 | relation默认关闭 |
| OT-07 | 9.8 | U的feature目标不依赖伪标签，logit/proto仅高共识U | `code/SSDG/train_ssdg.py` | verified | U函数无label/pseudo输入且反传测试通过 | 不改变FastTrust路由选择集合 |
| OT-08 | 9.19.1 | 默认三教师视图`clean+medium+hard` | `code/cvsrffi/deployment_orbit.py`、`code/SSDG/train_ssdg.py` | verified | 默认配置与真实调用计数测试通过 | 用户当前明确修订 |
| OT-09 | 9.19.2 | Temporal Orbit Memory效率实现 | `code/cvsrffi/orbit_teacher.py` | verified | 更新、查找、保存、恢复测试通过 | 仅A8默认使用 |
| TG-01 | 9.9、9.10 | 无量纲信道坐标和部署协方差联合方向 | `code/cvsrffi/deployment_orbit.py` | verified | 稀疏协方差方向及8类单参数方向的确定性、稀疏度和单位范数测试通过 | 联合训练方向每样本最多3个活跃坐标 |
| TG-02 | 9.11～9.14 | 全部署区域基点、参数分型、共同随机数、低维物理基 | `code/cvsrffi/deployment_orbit.py` | verified | 固定received-IQ基点、无新增随机信道、8类方向重放测试通过 | clipping/quantization未进入tangent基 |
| TG-03 | 9.15 | 单边有限差分训练及`delta`审计 | `code/cvsrffi/selective_tangent.py` | verified | 角度灵敏度、正`delta`负测通过 | 初始`delta=0.05` |
| TG-04 | 9.16 | 强tangent只施加在最终`z_id` | `code/cvsrffi/selective_tangent.py`、`code/SSDG/train_ssdg.py` | verified | 损失只读取`z_id`且主链反传通过 | 不强迫共享骨干完全不变 |
| TG-05 | 9.17 | Fingerprint Tangent保留发射机方向灵敏度 | `code/cvsrffi/selective_tangent.py` | verified | 指纹干预、下界损失及`R_select`测试通过 | 输出`R_select` |
| NB-01 | 9.16 | `z_dom`异方差预测信道状态 | `code/model_dual_cvsincnet.py` | verified | 9维均值/对数方差、NLL和旧checkpoint兼容smoke通过 | 默认关闭，不改变旧state dict |
| LS-01 | 9.18、9.21 | 联合损失、建议初值及辅助尺度控制 | `code/SSDG/train_ssdg.py` | verified | 联合损失、逐项EMA归一化、checkpoint恢复及共享身份骨干梯度比测试通过 | 梯度比仅作诊断，不形成自动gate |
| SC-01 | 9.20 | A/B/C/D课程调度 | `code/cvsrffi/orbit_teacher.py`、`code/SSDG/train_ssdg.py` | verified | 全部epoch边界测试通过 | E1～20、21～60、61～140、141～200 |
| DG-01 | 9.22 | dispersion、单参数灵敏度、`R_select`、`N_eff`、最差桶及资源 | 上述模块与训练输出 | verified | 8类单参数、6类分桶、dispersion、`R_select`、`N_eff`及既有资源summary均已接入 | 数值结果必须由真实实验产生，缺失元数据显式记为`NaN` |
| AB-01 | 9.23 | A0～A8及z/logit/proto/relation消融 | `code/cvsrffi/deployment_orbit.py`、launcher | verified | A0～A8、FastTrust-EFF入口和`no_z/no_logit/no_proto/relation_on`配置测试通过 | relation仍默认关闭 |
| FM-01 | 9.24 | 六类失败模式的可观测防护 | 上述模块 | verified | clean权重、`N_eff`、dispersion、共识率、逐项尺度、梯度比及灵敏度均可观测 | 阈值由真实训练校准，低性能不作技术停止条件 |
| PR-01 | 项目协议 | source-only、V只读、target/query不可达 | `code/SSDG/train_ssdg.py`、tests | verified | U路由签名负测和真实checkpoint无query smoke通过 | smoke记录`query_inputs=0`、`target_inputs=0` |
| BC-01 | 兼容要求 | 新方法关闭时旧行为、参数和checkpoint兼容 | 主链及模型 | verified | opt-in默认值、旧checkpoint装载及相邻回归通过 | 新功能全部显式opt-in |

## 实施顺序

1. 先为`deployment_orbit`、Orbit Teacher和Tangent纯函数建立失败测试。
2. 实现可重放的信道状态与三教师视图构造。
3. 实现鲁棒球面聚合、四层损失和Temporal Memory。
4. 实现选择性tangent、fingerprint keep和nuisance异方差头。
5. 接入`train_ssdg.py`，保持FastTrust-EFF选择集合不变。
6. 增加A0～A8配置、诊断和checkpoint状态。
7. 运行聚焦测试、协议负测、真实checkpoint无query smoke和一次P0/P1审查。

## 本地验证记录

- 解释器：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，对应项目要求的`ssr-gpu`环境；本机`conda activate`入口不可用，因此使用同一环境的解释器绝对路径。
- 聚焦与相关回归：`71 passed`，其中DAOT聚焦测试51项；覆盖纯函数、主链labeled/U路由、8类单参数诊断、逐项尺度、梯度比、独立损失消融、旧行为及Phase1详细评估。
- 真实checkpoint无query smoke：使用既有FastTrust E200 checkpoint，状态`PASS`；旧权重只缺预期的4个新增nuisance-head参数，`z_id`形状为`[2,160]`，nuisance均值形状为`[2,9]`，`query_inputs=0`、`target_inputs=0`。
- 语法验证：DAOT四个模块、模型、公共构建器、`train_ssdg.py`及smoke入口均通过`py_compile`。
- 相邻旧测试曾单独出现4个非本次回归：2个因仓库根入口找不到未修改的`train_fjmp`，2个仍期待旧`0.10/0.70/0.20`划分；当前HEAD本身已采用`0.08`默认和`rho_label<=0.1`协议约束，本次差异未修改这些行。
- Git Bash本地通道被桌面适配器错误路由到WSL并返回`ERROR_PATH_NOT_FOUND`，因此两个`.sh`仅完成静态配置测试，未声称本地Bash dry-run通过。
- P0/P1定点审查修复两项入口问题：真实checkpoint smoke缺少`code`模块路径；A0～A8矩阵未显式绑定FastTrust-EFF。修复后对应回归均通过。N607发布前仍需一次独立P0/P1审查。

## 交付判定

只有状态为`verified`的条目可称为已完成。`partial`表示核心路径已实现，但报告要求的周期诊断、损失尺度校准或真实N607结果尚未闭合；不得以单元测试代替实验结论。
