# OPGAC-Net冻结主干后置轻型网络实现报告

## 基本信息

- run_id：`opgac_net_impl_20260630_2305`
- 时间：2026-06-30
- 操作目标：按`天基RFFI冻结主干轻型网络设计报告.md`实现OPGAC-Net第一版，覆盖旧类域适应、新类少样本注册、未知类拒识与歧义判决。
- 协议依据：已读取`AGENTS.md`和`项目.md`。实现遵守Stage2-B/C边界：query不参与训练、上下文估计、阈值拟合、原型更新或回滚验证；unknown query只允许最终评估。

## 实现范围

| 设计模块 | 实现位置 | 状态 |
|---|---|---|
| 固定特征变换`T/PCA/whitening` | `code/cvsrffi/opgac_net.py`：`FixedFeatureTransform`、`fit_fixed_feature_transform` | 已实现 |
| 地面多原型高斯记忆库 | `GaussianClassState`、`OPGACMemory` | 已实现，支持每类多成分 |
| 目标域上下文编码器`Cφ` | `DeepSetContextEncoder` | 已实现，support-only |
| RF条件分支`Bφ` | `RFConditionBranch` | 已实现为可选轻量分支 |
| 低秩特征校准器`Aφ` | `LowRankFeatureCalibrator` | 已实现，context驱动、quality gate、残差归一化 |
| 旧类高斯校准器`Gφ` | `OldGaussianMemoryCalibrator` | 已实现，多成分逐成分均值/方差/阈值校准 |
| 新类少样本高斯生成器`Nφ` | `NewClassGaussianGenerator` | 已实现，单高斯新类注册、旧类协方差先验、重叠provisional |
| 能量拒识头`Rφ` | `EnergyRejectionHead` | 已实现，old/new/unknown/ambiguous判决 |
| memory-only回滚/漂移告警 | `rollback_memory`、`drift_alarm` | 已实现 |
| 现有评估兼容薄接口 | `register_old_classes_opgac`、`register_new_classes_opgac`、`predict_with_opgac_head` | 已实现，输出`PredictionResult` |

## 子agent监督结论

| agent | 结论 | 已采纳处理 |
|---|---|---|
| 协议监督 | query不得进入fit/calibrate/context/threshold/update/EMA/rollback_validation；`B_T`和online EMA只能作为diagnostic | `OPGACNet.initialize_memory()`只接收support；`predict()`无状态；未实现query EMA |
| 代码集成监督 | 最小侵入应兼容`PrototypeSet`、`ClassState`和`PredictionResult`；ambiguous不能计入unknown rejection | 新增`predict_with_opgac_head()`和`opgac_to_prediction_result()`；ambiguous映射为`uncertain` |

## 变更文件

| 工作区 | 文件 | SHA256 |
|---|---|---|
| 非Git主工作区 | `E:\type10-7\code\cvsrffi\opgac_net.py` | `096B95949BC6393ADC9C6F2A1EF57D173F1CD7685EAB4B1BC4200EDBC60451D8` |
| 非Git主工作区 | `E:\type10-7\code\tests\test_opgac_net.py` | `657DF11BCA487EEF3E99D7E67F4019B58F91421564A629B997DD93D13B1A6EC9` |
| Git-backed发布仓库 | `E:\type10-7\github_publish\CVS-RFFI-repo\code\cvsrffi\opgac_net.py` | `096B95949BC6393ADC9C6F2A1EF57D173F1CD7685EAB4B1BC4200EDBC60451D8` |
| Git-backed发布仓库 | `E:\type10-7\github_publish\CVS-RFFI-repo\code\tests\test_opgac_net.py` | `657DF11BCA487EEF3E99D7E67F4019B58F91421564A629B997DD93D13B1A6EC9` |

非Git主工作区快照：

- `E:\type10-7\code\snapshots\opgac_net_impl_20260630_2305\code\cvsrffi\opgac_net.py`
- `E:\type10-7\code\snapshots\opgac_net_impl_20260630_2305\code\tests\test_opgac_net.py`

## 验证命令

主工作区：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\cvsrffi\opgac_net.py code\tests\test_opgac_net.py
$env:PYTHONIOENCODING='utf-8'; C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_opgac_net.py -q
```

结果：`10 passed`。仅有`.pytest_cache`写入权限警告，不影响测试。

Git-backed发布仓库：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\cvsrffi\opgac_net.py code\tests\test_opgac_net.py
$env:PYTHONIOENCODING='utf-8'; C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_opgac_net.py -q
```

结果：`10 passed`。

## 当前边界

- 本次未启动N607实验，未远端同步。
- 本次实现是模块级和薄接口级落地，未默认改动现有`eval_spaceborne_fewshot.py`正式runner路径。
- 若要跑正式Stage2-C矩阵，需要新增显式`--method opgac`或等价candidate字段，并在score table中追加OPGAC列，不能替换既有指标口径。
