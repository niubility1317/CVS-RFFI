# ADV3B02 Phase2-C闭式ridge少样本头验证

## 运行身份

| 字段 | 值 |
|---|---|
| Run ID | `phase2_adv3b02_ridge_head_k1k5k10_no_unknown_20260703_1140` |
| 时间 | 2026-07-03 11:40 Asia/Hong_Kong |
| 操作方 | Codex |
| 目标 | 在Phase2-C中用目标域旧类support和新类support训练闭式岭回归分类头，验证K=1/5/10是否能同一行达到旧类>=85%、新类均值>=80%、每新类>=85%，未知类不参与 |
| 基础模型 | `ADV3B02_CORE90_SOFT_E200` |
| 部署约束 | support只用于闭式求解权重；星上保存权重矩阵和类别表，不保存原始support样本或support embedding |
| 最新补充口径 | 5-shot性能相对1-shot可允许下降3pp，但完成仍需同一行满足绝对门槛 |

## 方法

| 模块 | 设计 |
|---|---|
| 特征 | 冻结ADV3B02的`z_id` |
| 训练 | 对K-shot support执行闭式ridge：`W=(X^T X+lambda I)^-1 X^T Y` |
| 推理 | `argmax([z,1]W + old_bias)` |
| 存储 | 每个类别一列权重；不保存support |
| 训练开销 | 小矩阵求逆，适合星上轻量校准或地面生成后上注 |

## 本地变更与验证

| 文件 | 状态 |
|---|---|
| `E:\type10-7\code\scripts\phase2_ridge_support_head_sweep.py` | 已创建，SHA256=`3798E84AA4A09916D46B0AFE23B8C3CD6D349CA7178416E82E75D2857A8C8E52` |
| `code\tests\test_phase2_ridge_support_head.py` | Git-backed测试已创建 |

| 验证 | 结果 |
|---|---|
| RED | 新测试先因`ModuleNotFoundError: phase2_ridge_support_head_sweep`失败 |
| GREEN | `conda run -n ssr-gpu python -m pytest -q code/tests/test_phase2_ridge_support_head.py code/tests/test_phase2_compressed_proto_knn.py`通过，5个测试通过 |
| 语法 | `conda run -n ssr-gpu python -m py_compile code/scripts/phase2_ridge_support_head_sweep.py`通过 |

## 远端验证计划

| 项 | 值 |
|---|---|
| 远端脚本 | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_ridge_support_head_sweep.py` |
| 候选池 | `7-14`近边界候选、`7-7/8-8`新类较强候选 |
| K | `1,5,10` |
| 扫描 | `l2_grid=0.001..10`，`old_bias_grid=0..0.2` |
| 成功判据 | `old_acc>=0.85`、`seen_new_acc>=0.80`、`min_seen_new_class_acc>=0.85` |

## 当前状态

| 项 | 状态 |
|---|---|
| Git提交 | 待完成 |
| 远端同步 | 待完成 |
| 远端运行 | 待完成 |
| 结果解析 | 待完成 |
