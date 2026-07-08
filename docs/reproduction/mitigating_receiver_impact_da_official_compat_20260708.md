# Mitigating Receiver Impact DA官方兼容诊断

## 官方代码状态

已找到官方仓库：`https://github.com/YannLeo/Cross_Receiver_RFFI_Network`。README声明其为`Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation`的official implementation，但仓库只公开了`mine_pseudo_classweight_trainer.py`，缺少WiSig/ManySig配置、主入口、数据集处理和`MINE`模型注册，不能直接原样复跑Table II。

## 本地差距定位

此前`Proposed`均值为52.04%，论文Table II均值为96.14%，差距为-44.10pp；`Source only`均值为55.52%，论文为59.45%，只差-3.93pp。因此主要问题集中在`Proposed`训练动态，而不是数据集整体不可用。

官方trainer暴露出的关键实现差异如下：

| 差异 | 本地新增兼容项 | 说明 |
|---|---|---|
| MINE用`ma_et`移动平均稳定目标，T步更新为`-0.5*loss_kl` | `--kl-estimator-mode mine_ma`、`--mine-update-scale 0.5` | 原本地路径是裸DV-KL |
| 每个epoch重建`pseudo_labels/predicted_labels`并按目标样本index写回 | `--pseudo-state-scope epoch` | 原本地状态跨epoch累积 |
| 伪标签阈值直接作用于`output_t`最大值 | `--pseudo-score-mode logit`、`--pseudo-threshold-mode official` | 官方`output_t`同时传给`CrossEntropyLoss`，语义更接近logits |
| class weight零预测类别置1，并在当前batch预测写入后计算 | `--class-weight-timing current` | 避免零计数类别产生极端权重 |
| 训练批次使用`zip(source,target)` | `--batch-pairing zip_min` | 原本地target循环复用 |
| 每epoch用target test loss保存best | `--target-model-selection target_loss_best` | 这是target-label model-selection diagnostic，不应作为严格无监督DA证据 |

## 使用方式

官方兼容路径可用以下简化开关启用：

```bash
python -m paper_reproduction.mitigating_receiver_impact_da.train \
  --config paper_reproduction/configs/mitigating_receiver_impact_da_manysig_paper_faithful.json \
  --run-table2 \
  --manysig-pkl <ManySig.pkl> \
  --methods proposed \
  --official-compat \
  --target-model-selection target_loss_best
```

`--official-compat`会启用`mine_ma`、official阈值、logit伪标签、epoch级目标状态、当前预测class weight、`zip_min`批次配对；如果未显式设置`source_pretrain_epochs`，则设为0，以贴近官方单循环训练。

## 验证

本地单元测试：

```bash
conda run -n ssr-gpu python -m pytest tests/test_mitigating_receiver_impact_da.py -q
```

结果：`26 passed`。
