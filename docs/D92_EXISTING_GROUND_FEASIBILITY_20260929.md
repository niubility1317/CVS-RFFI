# 既有地面汇总包可行性核查（2026-09-29）

## 结论与范围

现有四个正式 Phase1 模型均有已绑定 checkpoint 的 `int8_domain_class_center_lowrank_residual_radius_v2` 摘要。它只覆盖 **单位化 z_id 的 160 维、旧 6 类、15 个源域聚合单元**。可用量为冻结域类中心、每类跨域中心残差的 rank-3 表示，以及每域每类的 P90 余弦距离半径。没有 FFT96、原始 time/frequency/PA 分支汇总，没有类内或共享样本协方差，也没有新类统计。

本次仅读取本地既有元数据快照、配置与实现代码，未加载权重、源样本、逐样本源特征或摘要数组值，未重新导出地面统计，未读取 query 成绩。以下“已核实”指已有 readback 证据与实现一致；本次未重新连接远端核验文件当前状态。按要求窄查工作树 `docs/` 和 `E:/type10-7/docs/`，未找到 `D92_GROUND*` 文件，故以明确列出的元数据与代码为依据。

## 证据入口

- `E:/type10-7/local_artifacts/d92_upgrade_20260928/ground_payload_readback.json`：四模型 manifest、来源声明、数组 shape/dtype/bytes、文件字节；本次未运行其生成工具。
- `E:/type10-7/local_artifacts/d92_upgrade_20260928/fixed_phase1_branch_metadata_20260929.json`：VERIFIED 的 exact-loader 架构属性、strict 加载与冻结状态记录；当时为零 forward。
- `E:/type10-7/local_artifacts/d92_upgrade_20260928/fixed_phase1_branch_resolved_probe_20260929.json`：已解析训练/模型参数元数据。
- 当前协议：`E:/type10-7/项目.md` §5.3.1、§5.3.2。配置：`configs/cvs_d92_matched_20260927.json`。
- schema/codec：`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py:24`、`:29`、`:48`、`:299`、`:451`、`:573`；读取绑定：`code/cvsrffi/d92_ground_summary.py:106`。
- 来源核验：`tools/cvs_native_artifacts.py:24`、`:62`、`:89`；地面聚合生成代码：`tools/cvs_d92_matched.py:140`、`:165`、`:195`。仅审代码，不执行这些生成路径。

`tools/inspect_d92_ground_payload.py` 的检查路径会使用 `np.load`，因此本次复用既有输出，不以“只查 shape”为由重新读取数组。

## 实际保存的内容

特征 schema 为 `ADV3B02:z_id:unit_l2:160:v1`。旧类顺序固定为：

```json
["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
```

| 数组 | shape | dtype | 逻辑字节 |
|---|---|---|---:|
| core_q | 6 × 160 | int8 | 960 |
| core_scale | 6 | float16 | 12 |
| residual_basis_q | 6 × 3 × 160 | int8 | 2880 |
| residual_basis_scale | 6 × 3 | float16 | 36 |
| residual_coeff_q | 14 × 6 × 3 | int8 | 252 |
| residual_coeff_scale | 14 × 6 | float16 | 168 |
| radius_q | 15 × 6 | int8 | 90 |
| radius_scale | 6 | float16 | 12 |
| **数值特征与尺度合计** | | | **4410** |

registry/schema 元数据另占 670 B，含 `residual_rank:int16` 的 2 B，逻辑载荷总计 5080 B。若仅按 dtype 汇总数值数组，会把该 rank 元数据计入而得到 4412 B；这不是多出一项特征统计。本文沿 codec/loader 的八项数值数组定义记账。

域 registry 为字符串 `"0"` 至 `"14"`。第一个模型中心域为 `"4"`，其余为 `"3"`；14 个残差域排除各自中心域。源配置为 5 个 RX × 3 天，原生 `wisig_domain=rx_day`，经连续 domain label map 导出。代码链支持 RX×day 的域语义，但摘要自身没有 opaque handle 到具体 RX/day 的映射，不能从编号猜实际接收机或日期。

| 量 | 实际含义 | 可支持的判断与边界 |
|---|---|---|
| 域类中心 | 既有 source L_s 在各域、各旧类的单位 identity 聚合中心 | 固定旧类锚点或固定域漂移参考；不是新类均值 |
| rank-3 残差方向/系数 | 每类不同域中心相对固定中心域的残差 SVD | 跨域聚合中心漂移；不是逐样本类内协方差，也不是 paired phase 响应 |
| P90 半径 | 域类内部样本到其 normalized centroid 的余弦距离 90% 分位；离线 histogram bins=4096 | 合规已有类内标量统计；不是方差、有效样本数或协方差矩阵 |
| registry 与尺度 | 冻结类/域顺序、量化尺度、schema | 绑定和解码，不增加独立观测 |

实际 strict allowlist 不含：样本数量、一阶/二阶样本矩、类内对角或完整协方差、共享样本协方差、跨分支协方差、paired phase/time 响应、FFT 统计、全局 FP16 location/scale、BN 统计、逐样本 ID/embedding。协议允许某种可选字段，不代表本包拥有该字段。

## 文件字节与 checkpoint 对应

地面目录模板：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/20260927-phase2-cvs-d92-practical-manytx-m5-r01/cvs-daot-rc4-s<seed>/ground
```

每目录仅使用 `manifest.json` 和 `int8_domain_class_center_lowrank_residual_radius_v2.npz`。

| model seed | checkpoint SHA256 | NPZ B | manifest B | 文件合计 B | checkpoint 文件 B |
|---|---|---:|---:|---:|---:|
| 2026092701 | f7ea5064d56711c3173b16636a52af27018ba11a459f0205de950c134362e53b | 5508 | 2875 | 8383 | 15992872 |
| 2026092702 | 72f26413e495dcf82317898f8729caa3d35499bf0b41e9671ffdceab2c90924b | 5385 | 2877 | 8262 | 15992936 |
| 2026092703 | 51fadc3b4ad3cbf83de01700a4a109430e7a57b1e10cb147f0be5f01593f26bb | 5319 | 2872 | 8191 | 15992872 |
| 2026092704 | 690fed6ea2a6532d0ad978c93306f0f35b61f4a3a3fc4b1d6494f0e7547c74bb | 5485 | 2876 | 8361 | 15992936 |

四摘要文件共 **33197 B**；数值逻辑量 17640 B、含 registry/schema 的逻辑量 20320 B。四个既有 checkpoint 包共 **63971616 B**，与摘要分列；这些是现有文件大小，不是已优化的最小推理包大小。仅当接收端确已部署相同摘要时，新增传输才可记 0 B；本地 readback 证明包存在，不能单独证明卫星端已部署。

## 来源与继承契约

四个 checkpoint 来源目录模板：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01/cvs-daot-rc4-s<seed>/final_ssdg.pth
```

既有记录一致显示 `scratch_only=true`、`checkpoint_sources=[]`、`target_training_contact=false`、`target_contact=false`、`source_roles=EXACT_MATCH`、`ema_origin=this_run_student`；完成状态 `TRAINING_COMPLETE`，200 epochs，选用 `final_ssdg.pth`，`target_evaluated=false`。训练 commit 为 `9b1ddd939fcabc74c77c94a62fef8e5aad07c0e9`，方法版本 `DAOT_A1_FastTrust_RC4_original_residual_noeq`。resolved config 为 from-scratch、`checkpoint_selection=final_only`、6 类、`equalized=1`；无继承旧 checkpoint/teacher。

源数据契约：RX 数据集索引 `[1,3,4,6,8]`，days `[1,2,3]`；L_s/U_s/V 比例 0.07/0.63/0.30，train_ratio=0.1。地面汇总使用 exact L_s，配置记录为全部 6300 条 L_s。`tools/cvs_native_artifacts.py` 检查实际 `role_ids/source_rxs/source_days/ratios/split_seed/num_classes` 与 source contract 一致，并核对 final-only 和目标接触禁令。引用的 `phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json` 是数据角色参照，不是旧权重继承。

四 manifest 均为 `cvs.matched.ground.v2`，声明 `CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L`、`formal_phase2_eligible=true`、`source_role=L_s`、`target_access=false`、`component_state=CURRENT_MATCHED_SOURCE_ONLY_V2`。每个 manifest 绑定上表 checkpoint SHA 及对应 source prototype artifact SHA：

| model seed | source_prototype_artifact_sha256 |
|---|---|
| 2026092701 | 8a46a2be31cd2b03ac2b7fa365870d313b1bf5714d90f2629463578c7b1d794b |
| 2026092702 | 4d5cd4ec9c2247f9838440311bee37d36dbf3e5fd0c3db40fc306d276dfae9c0 |
| 2026092703 | 88899b10869d96f29e697924013167f155957c9de8007b68a4ccfef8c360a8ec |
| 2026092704 | 0f3ee0c70de37fe910bca717c5fae1783269b1f8c6d795c96ae122116df3857c |

该 artifact hash 是来源引用。它不授权 Phase2 打开原 `source_l_features.npz`，也不授权重新构建 ground。已有 exact-loader 元数据核实 strict 199 tensors、missing/unexpected/skipped 均为 0、15 domains、256 点输入、eval/frozen、BN 数量 0；本次没有重新加载 checkpoint。

实际模型为 `lite_d/no_dac/feat_joint`，time/frequency/PA/stats 启用、DAC 禁用，相关 embedding 宽度 160。**模型能导出某个分支，不等于摘要已保存该分支统计。** 当前地面包不能为 736 维 branch 特征补出协方差或域漂移。仅对 identity 子块加固定先验也应明确只是 160 维局部参考，不能称完整 branch 分布先验。

## 使用权限与下一步边界

1. **普通原型属于 §5.3.1。** 只能作不可训练类别锚点或冻结判决；不能在线把它当训练样本拟合协方差、LDA 或持久分类头。
2. **本次实际 v2 包属于已绑定的 §5.3.2 量化摘要。** 在 checkpoint、类顺序、schema 与来源绑定保持有效时，可按预先固定规则用作冻结分布锚点、协议允许的确定性虚拟特征点或 normalization。不能将该例外扩大成源样本回放、伪源样本生成、逐样本源 feature 使用或持久反量化 bank。本文不提出虚拟样本训练方案。
3. 可从既有域类中心构造类无关的固定 identity 域漂移参考，再与合法 support 学习的状态按预登记公式结合。但这一量只是旧类源域聚合漂移；迁移到新类是建模假设，尤其不能声称解决 K1 的跨物理样本类内协方差不可辨识性。
4. 该机制已有实现先例：`stage2_d89_v2_radius_cauchy_center.py:59` 至 `:74` 使用跨域中心残差及半径权重；`stage2_d92_summary_joint.py` 的冻结摘要算子使用等类加权域协方差与 `A=(I+160G/trace(G))^(-1/2)`。仅重命名或重新调用这些算子不构成新增信息机制。
5. 没有证据支持从本包获得 FFT、time/frequency/PA 的类条件或共享协方差。也不能将 P90 半径直接换称经验方差，或由 rank-3 域中心方向声称获得 paired view 噪声统计。此类量若成为必要输入，本次现有包不足，不能静默新导出源统计补齐。

当前可行结论是：**已有合法小载荷可提供固定 identity 旧类/域漂移参考；不存在可直接供应完整 branch 度量的既有地面统计。** 当前下一候选若仅从本 row 合法 support 估计 scatter，无需依赖地面包。本核查不改变其公式、参数或实验范围，也不提供任何 query 性能判断。
