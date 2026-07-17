# Phase1域×类int8原型特许路线追溯

日期：2026-07-17

## 用户特许与边界

用户明确允许把地面Phase1训练形成的域×类压缩原型上传到星上，用于缓解源域/clean样本需求和部署存储成本。上传物只能是压缩后的聚合原型特征，不能包含原始IQ、样本级原始/全精度feature、exemplar、source cache或可逆样本索引。

该特许被限定为`sealed_phase1_deployment_bundle_with_optional_int8_domain_class_prototypes_v1`：ADV3B02 checkpoint与int8原型组件在任何target数据可达前共同生成、登记和整体封存。Phase2只读该组件，可临时解量化计算，但不得持久化全精度副本、更新/替换原型、执行source replay或使用query标签/角色/类别数/quota/global assignment。

## ADV3B02现存artifact取证

- checkpoint：`best_joint_safe_ssdg.pth`
- SHA256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- 大小：8,582,116B
- 顶层包含`model`、`ema_model`、训练参数、split与metric状态；`model`与`ema_model`各195个state entry。
- 训练参数记录`use_proto_memory=true`、`phase2_export_prototypes=true`、`phase2_fuse_prototypes=true`，但checkpoint的模型/EMA state中没有prototype、centroid、anchor或bank张量。
- 同一报告目录未发现`phase2_zid_prototypes.pt/.json`或其他可部署原型组件。
- 当前`sealed_feature_runtime.pt`是`ADV3B02IdentityRuntime` TorchScript，SHA256为`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`，未暴露原型buffer。

因此，现存ADV3B02不能直接“量化checkpoint内已有原型”。后续必须在Phase1/offline边界从授权训练split生成域×类多样本聚合质心，量化后与原checkpoint组成新的整体deployment bundle并重新封存；不得在Phase2临时回源。

## 固定量化格式

每个域×类原型`p`使用逐向量对称int8量化：

```text
scale = max(abs(p)) / 127
q = clip(round(p / scale), -127, 127)
```

部署payload仅允许：

- `int8[D,C,P]`聚合质心；
- 逐域×类FP16 scale；
- 固定domain/class registry；
- feature schema/version和不可变性证据。

禁止payload：原始IQ、单样本feature/logit、全精度prototype、exemplar、source sample ID/坐标、covariance、BN/Fisher/gradient、teacher/cache和可逆成员信息。

## 首个方法候选

首个开发候选为`D19-CIAF`：Checkpoint-Embedded Int8 Domain-Class Anchor Fusion。Phase2用合法固定LEO_weak target support形成target原型，再以support-only相容度选择/融合只读地面域原型；Stage2-C冻结全部旧类融合状态，仅追加新类稳健原型，并由support-only碰撞margin限制新类对旧类的侵入。

开发门禁必须同时覆盖旧类floor、新类floor、注册前后旧类遗忘、`H_old_new`和资源状态；任一场景/新类规模不满足逐类非退化时原子回退，不允许逐类或逐场景机会性保留。

## 资源硬约束

本路线继续执行更紧的正式上限：adapter可训练参数不超过50,000、适配不超过20epoch、无dense query图、持久化适配状态不超过256KB。int8原型组件计入部署状态量，但不计为可训练adapter参数；必须单列其payload、scale/registry、一次性enrollment MAC、每query额外MAC、平均/P95时延与峰值显存。
