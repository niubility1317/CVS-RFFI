# D11 完整 support 日志复核

2026-09-30。**D11 并非完全没有旧类适应收益：最终统一候选在 clear/rain 的同折旧类持出均值分别提高 3.33/5.00 个百分点，low_elev 为零；但追加新类后的旧类下降更大，分别为 8.33/15.00/6.67 个百分点，抵消了这些收益。** 这同时暴露了有限且不均衡的旧类适应，以及状态冻结后仍存在的注册竞争问题。不能将失败概括成“梯度没有工作”，也不能仅凭平均适应增益宣布顺序目标已经解决。

## 1. 输入与完整性

只读取原目录 `E:/type10-7/automation_reports/CV-SincNet/d8_second_block_dev_20260717_020200/d11_trainable_lowrank_rank8_v6_joint` 下两份已明确授权的完整 support 产物：

- [support_audit.json](E:/type10-7/automation_reports/CV-SincNet/d8_second_block_dev_20260717_020200/d11_trainable_lowrank_rank8_v6_joint/support_audit.json)。
- [training_log.jsonl](E:/type10-7/automation_reports/CV-SincNet/d8_second_block_dev_20260717_020200/d11_trainable_lowrank_rank8_v6_joint/training_log.jsonl)。

按 training-log-analysis 技能完整解析 1,038 个非空 JSON 对象；没有抽样、跳过行、读取 query、加载权重或重新运行模型。递归检查所有数值有限。日志 SHA256 与原 audit 自带字段一致：`cf0bb2cceeac2517e6ca7a7da647ea960728558d85f59e3c0ea5391fbe52b9d9`；这里只复核已有绑定，不建立新的签名或审批链。

| 阶段 | 记录数 | 训练轨迹数 | 完整性 |
| --- | --- | --- | --- |
| selection | 540 | 45 | 每轨迹 epoch 1–12；含标记 unified_hyperparameter_refit 的 180 行 |
| full_fit | 108 | 9 | 每轨迹 epoch 1–12；含统一重拟合的 36 行 |
| joint_registration_l2o | 360 | 30 | 2 候选 × 3 场景 × 5 fold × 12 epoch |
| joint_registration_fold_summary | 30 | 不适用 | 每候选/场景/fold 恰好 1 条，epoch=12 |
| 合计 | 1038 | 84 条训练轨迹 | 1008 条 epoch 记录 + 30 条 joint summary |

统一重拟合记录按其显式标记独立计数，没有作为重复行删除，也没有当作额外独立泛化重复。84 条轨迹都完整包含 epoch 1 至 12。1,008 条训练记录逐行重算四项加权损失，总损失最大绝对差为 3.36e-07，符合 float32 记录舍入。

audit 标记 query package/prediction/score/truth、scorer、source sample/derived signal access 均为 false。其旧 preopen 同时保留 `LOCAL_PROTOCOL_REPAIR_REQUIRED`、`STRUCTURAL_ONLY_REAL_INPUT_RECOMPUTE_REQUIRED` 和 formal_metric_claim_allowed=false。因此本报告是历史 support 机制复盘，不把旧产物升级为当前干净正式实验，不继承其权重、配置或旧门槛。未读取额外 stdout、checkpoint、NPZ 或输入包；无法据这两份文件独立重做物理数据和模型来源审计。

## 2. 配对口径与实际参数

K10 的 joint leave-two-out 每类取 8 条旧 support 训练 adapter，新增类用每类 8 条 support 注册，旧/新每类各 2 条持出。旧类 6 个、新类 5 个。三场景分别为 leo_clear_weak、leo_low_elev_weak、leo_rain_weak。记录中的 receiver=1-20、run seed=713201；训练 seed 是下表的 20260717，不能把不同 seed 字段混为一个。单一 base received view 不增加物理 K。

| 口径 | 旧类 | 新类 | 合计 |
| --- | --- | --- | --- |
| 注册类别数（原逐类字段计数） | 6 | 5 | 11 |
| 完整 K10 support 池记录数 | 60 | 50 | 110 |
| 每个 joint fold 的训练/注册 support | 48 | 40 | 88 |
| 每个 joint fold 的 held support | 12 | 10 | 22 |

样本数由 audit 中的 K10、6/5 个逐类键及明确的 old_k8_fit_new_k8_register 策略计算；60 条 before 和 110 条 after feature provenance 数量与其一致。Stage B 只使用旧类 48 条训练，Stage C 用这些固定旧状态和新类 40 条注册。这里的 48/40/12/10 不是日志显式独立 count 字段，也不是本次打开物理样本重新核验的结果；缺少 fold 物理 ID 清单的限制保留。

| 候选 | rank | learning_rate | temperature | prototype_weight | supervised_contrastive_weight | identity_weight | factor_weight | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| floor_seek | 8 | 0.02 | 0.1 | 1.0 | 0.25 | 5.0 | 0.02 | 20260717 |
| identity_strong | 8 | 0.005 | 0.1 | 1.0 | 0.25 | 20.0 | 0.05 | 20260717 |

所有轨迹 12 epoch。floor_seek 是历史最终统一选择，但原 status 为 SUPPORT_ONLY_D11_NOT_SELECTED_NO_QUERY_OPEN；selected_candidate_id 不表示通过晋级。下文 A0=identity-old，B=adapter-old，C=registered-old/new。A0 是**未加 adapter 的同 support 原型分类对照**，并非冻结地面 Stage A/DG，因此不能把 B−A0 直接写成用户真实 Stage B−Stage A。

最终候选的四种准确率均直接存在 audit 的同一 joint fold 中。对另一候选，identity-old 读取同场景 before_selection.identity_baseline 的 fold 原值，adapter-old 读取该候选 joint_registration_l2o 的 epoch12 loo_overall_accuracy，registered old/new 读取该候选同 fold 的 summary；同时核对日志已有两种 forgetting 差值，无任何从差值反推缺失准确率。最终候选 15 折的准确率及全部逐类字段与 audit 完全相符；30 折的差值恒等式全部相符。

这两份产物未保存每一 fold 的 held/train 物理 ID 清单；“同 fold”对应的是原程序所记录的同场景 fold 身份和字段一致性，并非本次重新证明了逐物理 ID 配对。audit 中的 feature provenance 虽有物理 ID，但缺少其 fold 分配，不补算或猜填该映射。

## 3. 同候选、同场景的完整顺序结果

准确率单位为 %；差值为百分点。表中均值先在本场景 5 个 fold 内计算。注册下降=adapter-old−registered-old，正值表示下降；净变化=registered-old−identity-old。

| 候选 | 场景 | identity-old | adapter-old | registered-old | registered-new | 适应增益 | 注册下降 | 旧类净变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| floor_seek | clear | 71.67 | 75.00 | 66.67 | 52.00 | +3.33 | 8.33 | -5.00 |
| floor_seek | low_elev | 73.33 | 73.33 | 66.67 | 46.00 | +0.00 | 6.67 | -6.67 |
| floor_seek | rain | 73.33 | 78.33 | 63.33 | 60.00 | +5.00 | 15.00 | -10.00 |
| identity_strong | clear | 71.67 | 71.67 | 61.67 | 52.00 | +0.00 | 10.00 | -10.00 |
| identity_strong | low_elev | 73.33 | 73.33 | 66.67 | 44.00 | +0.00 | 6.67 | -6.67 |
| identity_strong | rain | 73.33 | 73.33 | 61.67 | 62.00 | +0.00 | 11.67 | -11.67 |

所有 30 个 joint fold 完整列出；不筛除无改善或明显退化的 fold。末列为 training_log.jsonl 的 epoch12 行/summary 行，可定位原记录。

| 候选 | 场景 | fold | identity-old | adapter-old | registered-old | registered-new | 适应增益 | 注册下降 | 原日志行 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| floor_seek | clear | 0 | 66.67 | 66.67 | 50.00 | 50.00 | +0.00 | 16.67 | 495/496 |
| floor_seek | clear | 1 | 83.33 | 83.33 | 75.00 | 50.00 | +0.00 | 8.33 | 508/509 |
| floor_seek | clear | 2 | 83.33 | 83.33 | 75.00 | 60.00 | +0.00 | 8.33 | 521/522 |
| floor_seek | clear | 3 | 58.33 | 66.67 | 66.67 | 50.00 | +8.33 | 0.00 | 534/535 |
| floor_seek | clear | 4 | 66.67 | 75.00 | 66.67 | 50.00 | +8.33 | 8.33 | 547/548 |
| floor_seek | low_elev | 0 | 66.67 | 66.67 | 58.33 | 60.00 | +0.00 | 8.33 | 632/633 |
| floor_seek | low_elev | 1 | 58.33 | 58.33 | 58.33 | 40.00 | +0.00 | 0.00 | 645/646 |
| floor_seek | low_elev | 2 | 75.00 | 75.00 | 50.00 | 70.00 | +0.00 | 25.00 | 658/659 |
| floor_seek | low_elev | 3 | 83.33 | 83.33 | 83.33 | 20.00 | +0.00 | 0.00 | 671/672 |
| floor_seek | low_elev | 4 | 83.33 | 83.33 | 83.33 | 40.00 | +0.00 | 0.00 | 684/685 |
| floor_seek | rain | 0 | 83.33 | 83.33 | 66.67 | 70.00 | +0.00 | 16.67 | 769/770 |
| floor_seek | rain | 1 | 75.00 | 75.00 | 66.67 | 70.00 | +0.00 | 8.33 | 782/783 |
| floor_seek | rain | 2 | 58.33 | 75.00 | 41.67 | 30.00 | +16.67 | 33.33 | 795/796 |
| floor_seek | rain | 3 | 75.00 | 83.33 | 75.00 | 70.00 | +8.33 | 8.33 | 808/809 |
| floor_seek | rain | 4 | 75.00 | 75.00 | 66.67 | 60.00 | +0.00 | 8.33 | 821/822 |
| identity_strong | clear | 0 | 66.67 | 66.67 | 58.33 | 60.00 | +0.00 | 8.33 | 84/85 |
| identity_strong | clear | 1 | 83.33 | 83.33 | 75.00 | 50.00 | +0.00 | 8.33 | 97/98 |
| identity_strong | clear | 2 | 83.33 | 83.33 | 66.67 | 60.00 | +0.00 | 16.67 | 110/111 |
| identity_strong | clear | 3 | 58.33 | 58.33 | 50.00 | 40.00 | +0.00 | 8.33 | 123/124 |
| identity_strong | clear | 4 | 66.67 | 66.67 | 58.33 | 50.00 | +0.00 | 8.33 | 136/137 |
| identity_strong | low_elev | 0 | 66.67 | 66.67 | 58.33 | 70.00 | +0.00 | 8.33 | 221/222 |
| identity_strong | low_elev | 1 | 58.33 | 58.33 | 58.33 | 40.00 | +0.00 | 0.00 | 234/235 |
| identity_strong | low_elev | 2 | 75.00 | 75.00 | 50.00 | 50.00 | +0.00 | 25.00 | 247/248 |
| identity_strong | low_elev | 3 | 83.33 | 83.33 | 83.33 | 20.00 | +0.00 | 0.00 | 260/261 |
| identity_strong | low_elev | 4 | 83.33 | 83.33 | 83.33 | 40.00 | +0.00 | 0.00 | 273/274 |
| identity_strong | rain | 0 | 83.33 | 83.33 | 58.33 | 70.00 | +0.00 | 25.00 | 358/359 |
| identity_strong | rain | 1 | 75.00 | 75.00 | 66.67 | 50.00 | +0.00 | 8.33 | 371/372 |
| identity_strong | rain | 2 | 58.33 | 58.33 | 50.00 | 70.00 | +0.00 | 8.33 | 384/385 |
| identity_strong | rain | 3 | 75.00 | 75.00 | 66.67 | 70.00 | +0.00 | 8.33 | 397/398 |
| identity_strong | rain | 4 | 75.00 | 75.00 | 66.67 | 50.00 | +0.00 | 8.33 | 410/411 |

对最终 floor_seek：clear 的 2/5 折旧类适应改善，其余 3 折不变；low_elev 的 5 折全部不变；rain 的 2/5 折改善，其余 3 折均值不变。注册后 clear 的 4/5 折、low_elev 的 2/5 折、rain 的 5/5 折下降。rain fold2 从 identity 58.33% 提升到 adapter 75.00%，注册后却降到 41.67%；这是本报告中最直接的“适应有益但注册抵消收益”的同折例子。

## 4. 逐类变化不能由总体均值替代

下面保留最终 floor_seek 的全部 6 个旧类。类简称为原物理 class token 的排序索引，不代表可迁移的语义类别或额外训练权重。

| 简称 | 原 class token |
| --- | --- |
| C1 | cls_12765bcd9d257085f42147cfafd8c4bc1f09797410b636bc6d9c10a9b062fe19 |
| C2 | cls_366fd4446e173eaa8beac15c6de68a4b9424c43b8c9e0cd2798ed4e4a7951a73 |
| C3 | cls_5dbeddf91eebc854d032d44bdf8c8be87eb92a5391aa5e8237b48aba46a63584 |
| C4 | cls_b70a251d1ea787d79197b43a9fff28705b8f7db178ebf57d5ed379cbe54b61cb |
| C5 | cls_c5b3efac869515a9ae536fbff087064b91386646222c70f68a0e30fc48fcccca |
| C6 | cls_db052f212b5bc541f3084d522360f85afe9e0144dcae66647711e29ab6ae0cd3 |

| 场景 | 类 | identity | adapter | registered | 适应增益 | 注册下降 |
| --- | --- | --- | --- | --- | --- | --- |
| clear | C1 | 90.00 | 100.00 | 100.00 | +10.00 | 0.00 |
| clear | C2 | 90.00 | 90.00 | 60.00 | +0.00 | 30.00 |
| clear | C3 | 90.00 | 90.00 | 90.00 | +0.00 | 0.00 |
| clear | C4 | 70.00 | 80.00 | 70.00 | +10.00 | 10.00 |
| clear | C5 | 80.00 | 80.00 | 70.00 | +0.00 | 10.00 |
| clear | C6 | 10.00 | 10.00 | 10.00 | +0.00 | 0.00 |
| low_elev | C1 | 90.00 | 90.00 | 80.00 | +0.00 | 10.00 |
| low_elev | C2 | 90.00 | 90.00 | 80.00 | +0.00 | 10.00 |
| low_elev | C3 | 90.00 | 90.00 | 90.00 | +0.00 | 0.00 |
| low_elev | C4 | 60.00 | 60.00 | 60.00 | +0.00 | 0.00 |
| low_elev | C5 | 80.00 | 80.00 | 70.00 | +0.00 | 10.00 |
| low_elev | C6 | 30.00 | 30.00 | 20.00 | +0.00 | 10.00 |
| rain | C1 | 100.00 | 100.00 | 90.00 | +0.00 | 10.00 |
| rain | C2 | 100.00 | 100.00 | 60.00 | +0.00 | 40.00 |
| rain | C3 | 90.00 | 90.00 | 80.00 | +0.00 | 10.00 |
| rain | C4 | 30.00 | 80.00 | 80.00 | +50.00 | 0.00 |
| rain | C5 | 80.00 | 90.00 | 60.00 | +10.00 | 30.00 |
| rain | C6 | 40.00 | 10.00 | 10.00 | -30.00 | 0.00 |

rain 的 C4 从 30% 到 80%，同时 C6 从 40% 到 10%；总体 +5pp 并不代表各旧类都受益。low_elev 总体适应增益为零，也不能由此断言所有逐类结果或逐样本预测相同。旧报告的逐类/floor 条件仅保留为历史说明，不提升为当前任务的新硬门槛。

## 5. 完整训练曲线核查

| 阶段 | 完整轨迹 | 末 loss 小于首 loss | 末训练准确率小于首轮 | 存在局部 loss 回升的轨迹 |
| --- | --- | --- | --- | --- |
| selection | 45 | 45 | 12 | 0 |
| full_fit | 9 | 9 | 2 | 0 |
| joint_registration_l2o | 30 | 30 | 6 | 0 |

损失是该步更新前记录，support/loo 准确率是更新后的重新评分；因此首行不是 epoch0 恒等准确率。下面逐折展示全部 joint 曲线的首末值、全 12 步扫描的最高 held 准确率及对应 epoch、首次低于同折 identity 基线的 epoch。最高值只作事后诊断，不据此重选 epoch 或重跑。

| 候选 | 场景 | fold | loss 1→12 | train acc 1→12 (%) | held acc 1→12 (%) | held 最高 (%) / epoch | 首次低于 identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| floor_seek | clear | 0 | 2.2490→1.1379 | 81.25→81.25 | 66.67→66.67 | 66.67 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| floor_seek | clear | 1 | 2.2269→1.1697 | 83.33→81.25 | 83.33→83.33 | 83.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| floor_seek | clear | 2 | 2.2788→1.2504 | 79.17→79.17 | 83.33→83.33 | 83.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| floor_seek | clear | 3 | 2.2257→1.1487 | 83.33→81.25 | 58.33→66.67 | 66.67 / 10,11,12 | 未出现 |
| floor_seek | clear | 4 | 2.2405→1.1630 | 81.25→81.25 | 66.67→75.00 | 75.00 / 11,12 | 未出现 |
| floor_seek | low_elev | 0 | 2.2609→1.1546 | 85.42→87.50 | 66.67→66.67 | 66.67 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| floor_seek | low_elev | 1 | 2.2529→1.1091 | 89.58→89.58 | 58.33→58.33 | 58.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| floor_seek | low_elev | 2 | 2.2895→1.2286 | 83.33→87.50 | 75.00→75.00 | 83.33 / 6,7,8,9,10,11 | 未出现 |
| floor_seek | low_elev | 3 | 2.2755→1.2226 | 89.58→87.50 | 83.33→83.33 | 83.33 / 1,2,3,4,5,6,11,12 | 7 |
| floor_seek | low_elev | 4 | 2.3160→1.2731 | 81.25→87.50 | 83.33→83.33 | 83.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| floor_seek | rain | 0 | 2.2053→1.1136 | 87.50→83.33 | 83.33→83.33 | 91.67 / 7,8,9,10 | 未出现 |
| floor_seek | rain | 1 | 2.1641→1.0669 | 83.33→85.42 | 75.00→75.00 | 75.00 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| floor_seek | rain | 2 | 2.1627→1.0335 | 91.67→85.42 | 58.33→75.00 | 75.00 / 11,12 | 未出现 |
| floor_seek | rain | 3 | 2.2073→1.1327 | 87.50→87.50 | 75.00→83.33 | 83.33 / 8,9,10,11,12 | 未出现 |
| floor_seek | rain | 4 | 2.2138→1.1178 | 91.67→85.42 | 75.00→75.00 | 75.00 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | clear | 0 | 2.2492→2.2003 | 81.25→81.25 | 66.67→66.67 | 66.67 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | clear | 1 | 2.2271→2.1788 | 83.33→83.33 | 83.33→83.33 | 83.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | clear | 2 | 2.2790→2.2350 | 79.17→79.17 | 83.33→83.33 | 83.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | clear | 3 | 2.2259→2.1775 | 83.33→83.33 | 58.33→58.33 | 58.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | clear | 4 | 2.2407→2.1923 | 81.25→81.25 | 66.67→66.67 | 66.67 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | low_elev | 0 | 2.2611→2.2161 | 85.42→85.42 | 66.67→66.67 | 66.67 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | low_elev | 1 | 2.2531→2.2085 | 89.58→89.58 | 58.33→58.33 | 58.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | low_elev | 2 | 2.2897→2.2471 | 83.33→83.33 | 75.00→75.00 | 75.00 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | low_elev | 3 | 2.2757→2.2302 | 89.58→89.58 | 83.33→83.33 | 83.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | low_elev | 4 | 2.3162→2.2750 | 81.25→81.25 | 83.33→83.33 | 83.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | rain | 0 | 2.2055→2.1593 | 87.50→89.58 | 83.33→83.33 | 83.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | rain | 1 | 2.1643→2.1144 | 83.33→83.33 | 75.00→75.00 | 75.00 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | rain | 2 | 2.1629→2.1128 | 91.67→91.67 | 58.33→58.33 | 58.33 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | rain | 3 | 2.2075→2.1611 | 87.50→87.50 | 75.00→75.00 | 75.00 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |
| identity_strong | rain | 4 | 2.2140→2.1658 | 91.67→91.67 | 75.00→75.00 | 75.00 / 1,2,3,4,5,6,7,8,9,10,11,12 | 未出现 |

上述现象足以检查训练目标与准确率是否同步，却不足以判定唯一因果原因。特别是 loss 下降可伴随训练 top1 下降，不能把所有问题都归因于“训练过拟合导致 held 降低”；原型 CE、SupCon 和正则组合优化的量不是 top1。日志没有梯度范数和逐步参数位移，不能从 gate_linf 单独判断某个低秩方向学到了什么。

## 6. 缺失字段、可支持结论与下一步

| 证据 | 本次状态 | 限制 |
| --- | --- | --- |
| 逐样本 score/logit、旧对新 margin、预测类别、混淆矩阵 | MISSING | 不能定位哪个新类吸走哪个旧样本，不能测量边界距离或校准 |
| 每折 train/held 物理 ID 映射 | MISSING | 仅核对原记录同 fold 字段，未独立重放 ID 配对 |
| 注册阶段每 epoch 的新类/旧类 held 曲线 | MISSING | 注册只保存最终 fold summary，不能推算注册下降最早发生在哪个训练 epoch |
| 真正地面 Stage A 同样本预测 | MISSING | identity support 原型对照不可冒充地面 A |
| 真实 K1/K5 持出结果 | MISSING/DEFERRED | 不能从 K10 切片或这批 K8 训练的折外推出真实 K1/K5 |
| 当前 Phase1/当前 D92 同分布复验 | 未执行 | 历史数值不能冒充现行方法性能，也不授权复用旧状态 |

依据既有源码，注册冻结 adapter 和旧原型，仅增加新类分数列；本 audit 也明确记录 bitwise lock=true。因此“新增竞争可以抵消旧类适应收益”是有机制依据的解释，而不是把下降归咎于注册阶段的梯度更新。可是两份日志不含逐样本 score/预测，无法从本次复核独立证实每一个翻转的目的类别、margin 大小或错误样本集合。不给缺失分数补零，不从总体差值伪造逐样本 margin。

对 Residual8 的直接教训：其小 rank、恒等起点、表示保持正则和冻结继承并非未尝试机制；必须分别看旧类持出增益及全注册竞争下的存留，不能只看训练 CE、参数量或 B 的总体准确率。新的 GELU、固定锚点和幅度约束尚无证据解决上述失败，且重新拟合 LocalRidge 还允许旧类分数改变。本复核不立即提出另一组 loss/rank/步数，也不根据历史最好 epoch 调参。

最小信息诊断现已完成到原产物可支持的边界。后续若在当前合法 support 上另行开展已授权诊断，最有用的补充是保存同一 held 物理 ID 在适应前、适应后和全注册后的完整类分数、预测及旧/新最高竞争 margin，并在保持原 head 状态与重新注册 head 两种明确状态之间标注变化来源。这是信息记录建议，不是新增实验、门槛或本次已经实施的对照。

本次只生成本文和 `.codex_tmp/recheck_d11_support_20260930.py` 标准库解析脚本；未训练、读取 query/权重、修改既有产物、执行 Conda/Git/远端操作，未改变当前健康实验。
