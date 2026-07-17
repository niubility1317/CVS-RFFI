# D22 ADV3B02 int8原型bundle最短inventory

日期：2026-07-17
范围：只读manifest/量化组件/class binding盘点；未打开query、truth或scorer，未运行实验。

## 1. 当前结论

当前D21复用的formal enrollment bundle**没有**Phase1 int8原型成员。其入口为：

`E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only\package_manifest.json`

该manifest的`phase2_pretrained_artifact_policy=sealed_phase1_checkpoint_only`，成员全集只有sealed runtime、method lock、overlay provenance和3个LEO_weak support文件。ADV3B02 runtime SHA256为`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`，对应Phase1 checkpoint SHA256为`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。

唯一可立即用于`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`的组件是D19历史研发副本；它未与checkpoint共同正式封存，状态为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`且`formal_phase2_eligible=false`。

## 2. 历史int8组件

首选压缩副本目录：

`E:\type10-7\automation_reports\CV-SincNet\d19_ciaf_int8_proto_20260717_1039\historical_int8_component_unverified_v2\`

|文件|schema/用途|大小|SHA256|
|---|---|---:|---|
|`int8_domain_class_prototypes.npz`|`phase1_int8_domain_class_centroids_v1` payload|5,363B|`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`|
|`manifest.json`|组件manifest|2,391B|`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`|

绑定hash：

- checkpoint：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- registry：`360e862e9e83c789d78e8ec2614a6ceda6b8392fe9c7851479579bc704428541`
- historical source aggregate：`e6ef79ce0c002539317efa79c1aac605ecb660003190c41b8a4481a9a4affcbd`
- historical deployment root：`5ed07c9fb72105f0ba9e9cf7c251fdc81fb22d9db42ab7dedb7d24c824b65ed0`

逻辑张量为`int8[26,6,160]`；有效domain行索引为`[0,1,4,5,8,9,12,13,16,17,20,21,24,25]`。14个有效域×6类=84个有效cell，占156个稠密槽位中的84个；每类均有14个有效cell。

|量化字段|shape/dtype|含义|raw-array SHA256|
|---|---|---|---|
|`domain_class_q`|`[26,6,160] int8`|逐域×类质心；当前唯一center表示|`ecc583df81bd63dad5ce72810ad835eb150b4a00b5fac6ba2992d4cbb81a6fcc`|
|`domain_class_scale`|`[26,6] float16`|逐向量对称量化scale|`bd976d8c195b9b9f37013607d70b628dde260e7f61acd9aeb39ef36aaa0a8255`|
|`domain_class_mask`|`[26,6] uint8`|有效cell mask|`0bbe62eaa76d94cece8cfa0273d4b8d4323df4f16a5f6ed6480f69ed53b3ff36`|
|`domain_registry`|`[26] int16`|域registry 0–25|`7229cccef885f824505d63d559d19c2fe2cb43186466f74a6d481a9634bf5594`|
|`class_registry`|`[6] str`|旧类TX顺序|`a314c2a542554da07731fd16b6904c9329455480f36e86e6ad37b518c63c9375`|
|`feature_schema`|scalar str|`ADV3B02:z_id:unit_l2:160:v1`|`45cf589c1f84679955ac726536a1c2968b18b2334f28d5e6241e4b0dee961f5c`|

有效payload为13,440B int8+168B FP16 scale=13,608B；当前稠密运行时状态为24,960B q+312B scale+156B mask=25,428B。

## 3. 旧类TX→handle映射

当前映射文件：

`E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json`

|class index|Phase1 TX|registered class handle|
|---:|---|---|
|0|14-10|`cls_75aa6d506081240f50cf3b79a0bd91714fa0084a635a472ca63194e57ec1dca2`|
|1|14-7|`cls_8b02d99905a8fe579368ac8e37eff51c505aaa89a646eba8892d5d800aa08416`|
|2|20-15|`cls_1f33441efa14970113b27483344b7df852a9041984b38d34ce570fafbab6689c`|
|3|20-19|`cls_f8dfc2edcccc5344f8e2535a959f13b53a1cddfd6fb22aed6e714de382b58d24`|
|4|6-15|`cls_a53ca1280d8fca58e3f4d6d1e9ddabfdab6027a941ee8c3f8c01d9d8ec945725`|
|5|8-20|`cls_33bbd16556c6e6305d1b7162f5ea71393afba910a922f9abca5999d5921a2d9d`|

当前文件SHA256为`bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f`。D19历史support screen锁定的是commit`1fcd9ab7`版本，SHA256为`4f701ac96a03761859810286a43f8576b0e642191be0fc0beb96d2c90785b5b7`；映射顺序相同，但当前文件增加了direct-logit行hash证据，因此重跑时必须使用当前文件hash重新预锁，不能沿用旧hash。

## 4. center/offset/radius状态

当前5,363B历史组件只有`domain_class_q+domain_class_scale`质心表示：

- center：有，字段`domain_class_q`；
- scale：有，字段`domain_class_scale`；
- offset/residual：无；
- radius：无。

代码中已有但尚未发现任何持久artifact的v2 schema：`int8_domain_class_center_lowrank_residual_radius_v2`，实现位于：

`E:\type10-7\github_publish\CVS-RFFI-repo\code\cvsrffi\phase1_center_lowrank_prototype_bundle.py`

其预定义字段为`core_q/core_scale`（固定center）、`residual_basis_q/residual_basis_scale`与`residual_coeff_q/residual_coeff_scale`（低秩offset）、`radius_q/radius_scale`，以及center/domain/class registry。radius定义为`p90_cosine_distance_to_phase1_domain_class_centroid`。当前工作区没有找到`int8_domain_class_center_lowrank_residual_radius_v2.npz`，所以没有可列的v2组件hash、radius生成proof hash或实际大小。

## 5. PRE_FORMAL support-only入口

入口存在：

- runner：`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\run_d19_support_only_ciaf.py`
- 固定mode：`development_select_unverified_component`
- int8组件：上述`historical_int8_component_unverified_v2`
- class binding：上述`d19_adv3b02_class_binding_20260717.json`
- support根：D18 capsule的`predictor/before/enrollment_only`与`predictor/after/enrollment_only`

已有无query执行receipt：

`E:\type10-7\automation_reports\CV-SincNet\d19_ciaf_int8_proto_20260717_1039\n607_k10_new5_rx20_1_seed713101\RECEIPT.json`

其状态为`DEVELOPMENT_SUPPORT_ONLY_COMPLETE`、`query_opened=false`、`formal_launch_authority=false`、`formal_metric_claim_allowed=false`；选择结果为`Z0_SUPPORT_ONLY`且`selected_positive_route=false`。因此入口可执行，但现有screen没有选出使用int8组件的正路线。

## 6. 最小缺口

1. D21正式deployment bundle未包含int8组件，仍是`sealed_phase1_checkpoint_only`。
2. 历史v1组件`formal_phase2_eligible=false`，缺checkpoint+int8共同新seal与新method lock。
3. center/offset/radius v2只有代码schema，没有NPZ、manifest、radius proof和hash。
4. 当前class-binding字节hash已从历史screen的`4f701...`变为`bb89...`，重跑前须重新锁定当前hash。
5. 既有PRE_FORMAL screen为负：`selected_positive_route=false`；不得据此打开query或进入正式矩阵。
