# CORE90部分证据与条件响应头实施报告

2026-09-11。实现位于独立分支`codex/core90-evidence-head-20260911`，基线为`340370900dfd9a0e92f24c535a24c7dec0b17980`。未覆盖主承载面的既有修改，未启动N607正式实验。

本轮完成H0–H5代码、真实模型集成、source-only拟合、support注册、冻结预测与独立评分。验证范围为数学参考、合成IQ和本机真实训练入口；不包含200epoch真实数据矩阵，不构成性能提升、物理可辨识性或晋级证据。逐项设计对应见[追踪表](core90_evidence_traceability.md)。

## 实现与消融

|版本|实际执行内容|按消融关闭的内容|
|---|---|---|
|H0|原CORE90分类头；同一工厂、导出和预测入口|全部新证据机制|
|H1|观测mask、对角高斯边缘似然、受控退化观测误差|低秩项、状态响应、support后验、类对专家|
|H2|H1加共享低秩相关协方差|状态响应、support后验、类对专家|
|H3|H2加共享/类差异条件响应、状态误差传播和域外诊断|support后验、类对专家|
|H4|H3加缺失异方差support后验及训练episode|类对专家|
|H5|H4加共享反对称类对残差和Laplacian全局投影|无；仅显式选择H5才开启专家|

`joint`使用实际160维联合表示；`blocks`使用时域、频域和PA局部块，共480维。部分观测先选择协方差子矩阵再求解，包含logdet和高斯常数；全部缺失返回无判别证据。块宽缩放固定，删除维度不会重新缩放剩余坐标。

原历史launcher参数、历史parser默认值及后续附加模块禁用项固定在`code/configs/core90_evidence_h0_h5.json`。保留CORE90的E80开始、0.68权重、CE-only卫星训练及原三场景课程；当前协议覆盖历史joint_safe选模和旧数据比例，采用从零初始化、source-V、单V与final-only。旧CORE90权重不会因为名称匹配而获得加载权限。

## 机制与参数验证

验证不只检查配置存在：H1–H5均执行真实CVSincNet前向、反向、优化步骤、快速路径一致性、逐样本/批量一致性及严格重载。均值、对角方差、低秩因子、H3以上响应斜率和H5专家检查有效梯度；参数更新后验证保存重载预测一致。U_s路径单独反向，确认不能更新证据头参数。

|参数/机制|运行证据或对照|
|---|---|
|variant、layout、covariance_rank|各版本实跑；H1非零rank、拼错字段及未校准模型均报错；480维块路径实跑|
|observation_floor/ceiling|改变参数使实际观测方差改变；质量模型只通过L_s受控退化拟合|
|response_variance|初始化可训练响应方差，并参与域外保守惩罚；有效方差与域外非零协方差受测|
|state_error_variance|改变状态误差buffer，传播协方差随之改变；H1/H2记录为消融关闭|
|prior_precision、support_weight|改变参数使实际support训练目标改变；缺失、相关、异方差后验对照完整联合条件化参考|
|pair_strength、pair_anchor|改变参数使最终scores改变；缺失坐标NaN/极值不得泄漏进专家|
|nll_weight、response_regularization|改变参数使实际总损失改变；辅助NLL对表示detach，防止该项单独驱动特征塌缩|
|partial_dropout|训练产生实际部分mask；显式观测mask优先，推理无随机dropout|
|原卫星路径|本机CUDA autocast执行三种实际`leo_*_weak`变换，并检查有限logits与反向梯度|
|epoch激活检查|检查观测误差、相关项与support调用；适用版本缺少support激活会明确失败|

零初始化或消融造成的未激活，与机制故障分开记录。`mechanism_manifest`输出配置、训练参数组及`inactive_by_ablation`；训练日志写入`evidence/*`统计。状态误差和专家在初始零贡献时不能仅凭配置宣称生效，实际更新和干预对照由测试覆盖。正式训练仍需读取全量实际日志判断持续激活。

## 训练与部署接口

入口为`code/scripts/run_core90_evidence.py`，提供`train`、`fit`、`predict`、`score`。训练必须显式提供数据、契约、source/target receiver与day，避免默认值猜测数据边界。可先用以下格式核对完整解析参数；占位符必须替换为本次真实值：

```text
python code/scripts/run_core90_evidence.py train --dataset DATASET.pkl --contract CONTRACT.json --source-rxs SOURCE_RXS --source-days SOURCE_DAYS --target-rxs TARGET_RXS --target-days TARGET_DAYS --variant H0 --output NEW_RUN_DIR --dry-run
```

实际运行删除`--dry-run`。`train`采用完整CORE90默认预算，成功后生成`ground_checkpoint.pt`和`deployment.pt`。前者保存数据契约/来源供地面核验；后者仅含模型构造字段与冻结模型状态。含NumPy RNG的本次训练checkpoint在内部可信边界剥离为tensor-only产物，公开fit/predict加载保持`weights_only=True`。

`fit`从同契约、无目标接触且从零训练的H0地面checkpoint开始，冻结同一backbone，只拟合新头。输入source张量必须恰好包含`x/y/physical_ids`，物理ID集合必须与契约L_s完全一致。该路径适合先比较同一表示上的H1–H5；原头、cosine、对角Gaussian、线性ridge的同表示对照由`FixedReadoutBaseline`提供。强对照的真实评估尚未运行。

数据契约包含`dataset_id`、`source_receivers`、`target_receivers`和`roles`，后者明确`L_s/U_s/V/target`物理ID列表。实际loader的dataset、receiver和物理ID逐一匹配；raw/equalized view不算新的物理样本。错误契约、未知上游、目标接触和非source选模均拒绝。

`predict`只读取received whitelist及可选support，要求既有`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`。H4/H5可由support注册旧类/新增类，输出全部注册类分数；后验不保留样本缓存，query不更新参数。H0支持原头预测；H0的support对照使用独立baseline API，不能把H4后验静默套在H0上。预测文件禁止覆盖；`score`另读truth，核对ID后计算指标，不把结果回流模型。

校准由`ConditionAwareCalibrator.fit(..., role='V')`显式完成，输入冻结模型在source-V上的logits、Mahalanobis、观测维数、quality、标签及状态覆盖。`state_dict()`可写入JSON交给`predict --calibrator`，重复拟合被拒绝。未提供校准文件不应宣称已完成选择性决策校准。分组样本不足或状态域外输出defer；不匹配只称候选，不能当作真正未知TX。评分同时提供拒绝计错准确率、闭集准确率、coverage/risk、NLL、Brier、ECE和risk-coverage曲线。

## 已修复的验证缺陷

1. 部分观测按剩余维度归一化会改变已观测坐标，现改为固定块宽缩放。
2. 专家原本可能读到缺失坐标对应的均值/方差，现对全部相关输入同步屏蔽并显式输入mask。
3. 状态clamp不能伪装成覆盖充分，现输出域外距离并保留保守方差惩罚。
4. 大维度FP32后验累加造成精度/对称性问题，现用FP64求解并对理论对称结果消除数值误差，仍拒绝真正不对称输入。
5. 实际训练checkpoint含NumPy RNG，现通过内部剥离支持安全部署重载。
6. H0及集成训练增加自动部署导出；目标物理ID核验改为对照实际loader。
7. 真实训练入口暴露非MUSE遥测变量未初始化，已修复。
8. 后续机制晋级规则误阻断CORE90，现以`core90_research`区分训练完成和科学晋级；实际checkpoint、请求的prototype文件、评估完成及既有协议检查保留，`promotion_ready=false`。

## 验证边界

本机`ssr-gpu`聚焦套件及相邻回归共157项通过，0失败、0跳过，用时28.729秒；包含实际CUDA三LEO测试。随后把L_s质量重校准移至V评估之前，保证V指标与保存状态一致；对最终变更复跑生产入口与profile，5项全部通过。CLI帮助及`git diff --check`通过。已有AMP弃用提示属于FutureWarning，未造成失败。

可检查JUnit证据：[157项聚焦与回归](../local_artifacts/core90_evidence_verification.xml)、[最终入口5项复测](../local_artifacts/core90_evidence_final_entry.xml)。不将两轮重叠用例相加为162项不同测试。

本轮合成生产夹具执行一个真实H5 epoch，验证optimizer/loss/log/checkpoint/导出/预测路径；不是完整训练。三LEO测试是合成输入的技术压力验证，不能代替真实clean/LEO/弱接收机性能、200epoch稳定性、训练时间和显存比较。

received state与quality是代理，既未证明等于PA发射激励，也未证明消除了RX身份。已提供source到V的TX/RX关联探针、匹配状态跨RX工具、rescue/harm、Schur信息和Fisher诊断；正式实验应据实运行这些工具，不能把其“已实现”写成“物理机制已验证”。状态误差默认值、域外惩罚和高斯假设属于待验证建模选择。

正式科学结论还需要合法数据契约下的完整H0–H5及强对照、source-only校准、全部预测后独立评分和多seed证据。本轮未读取目标真值、未加载旧污染权重、未进行性能驱动的选型或重跑。
