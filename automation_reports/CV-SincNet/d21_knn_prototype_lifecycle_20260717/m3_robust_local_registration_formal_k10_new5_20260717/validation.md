# M3验证记录

## 运行边界

- receiver/seed/K/new：`20-1/713101/10/5`。
- 输入仅读取3个已落地`LEO_weak`母缓存的`leo_weak_iq`，运行时硬检查`overlay_applied=true`、单场景、receiver一致。
- 与formal capsule逐场景核对：support IQ逐元素完全相等、support post-channel hash顺序完全相等、query post-channel hash集合完全相等。
- 固定`A0=z_id160+8*FFT96`，无特征或权重query调参。
- 五个候选都以support生成逐向量对称int8状态，并使用解量化状态实际完成LOO及query预测。
- `selector_lock.json`先于query评分写出；方法选择只依赖三场景support LOO，统一规则，不按场景、query角色或query类别配额选择。
- 15个当前prediction artifact仅含`query_token`与`predicted_class_index`；query truth未进入预测artifact。首次命名为`top2_mean`的3个旧artifact完整保留于`superseded_first_pass/`，不属于当前评分集合。

## 自动检查

- `ssr-gpu`下`python -m py_compile run_m3.py`：PASS。
- 当前prediction artifact数量：15（5方法×3场景）。
- prediction schema：15/15 PASS。
- formal capsule alignment：3/3 PASS。
- `query_used_for_selection=false`：PASS。
- 五方法int8状态均小于256KB；最大为top1/top2_trimmed的28,600B，预锁定bagged2为5,720B。
- adapter参数0、适配epoch 0、无dense query图。

## 文件哈希

- `run_m3.py`：`56630DE90DA599621C49A66536936FF900D866B21825316AE5DEEF8D6D85AFB2`
- `results.json`：`CC34D9DCD261B67BE535DAC8AD8DAAF17F76F7A9CEB9D55E79673A1663A53407`
- `selector_lock.json`：`FECAC88F550B3F168481F1AAA50DE20D5EF2C574125DEF43FD9F34C3D10BE271`

## 结论边界

这是单receiver×单seed的开发筛选。预锁定bagged2在formal query上未达到项目目标，尤其low-elevation/rain旧类floor分别只有20%/15%；因此只作为M3负向开发证据，不进入125确认矩阵或正式达标声明。
