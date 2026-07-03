# ADV3B02 Phase2-C压缩原型KNN低shot方法验证

## 运行身份

| 字段 | 值 |
|---|---|
| Run ID | `phase2_adv3b02_cproto_knn_k10_no_unknown_20260703_1125` |
| 时间 | 2026-07-03 11:25 Asia/Hong_Kong |
| 操作方 | Codex |
| 目标 | 在不保存原始support样本/逐support embedding的约束下，将KNN变体改造为压缩原型/子原型记忆库，并验证K=10是否能在Phase2-C同一行达到旧类>=85%、新类均值>=80%、每新类>=85% |
| 基础模型 | `ADV3B02_CORE90_SOFT_E200` |
| 未知类 | 不导出、不评估、不参与阈值或成功判据 |

## 方法变体

| 模块 | 设计 |
|---|---|
| 支持集使用 | support只用于一次性构建记忆库 |
| 星上保存 | 每类`1/2`个归一化原型、类内半径、样本计数、old/new标记 |
| 星上不保存 | 原始IQ、原始support样本、逐support embedding |
| 分类打分 | `score=max_m cos(z_q,mu_c,m)+old_bias-radius_weight*r_c,m` |
| 训练开销 | 无反向传播；只需ADV3B02前向提特征和轻量原型统计 |
| 部署适配性 | 类数扩展时只追加新类原型统计；存储量约为`类别数*子原型数*embedding_dim`，适合星上低开销注册 |

## 本地变更与验证

| 文件 | 状态 |
|---|---|
| `E:\type10-7\code\scripts\phase2_compressed_proto_knn_sweep.py` | 已创建，SHA256=`826A4C89F88C035B4AE5D664E94F4D4DE51910B81B8173049BB86FF520FDEBB3` |
| `code\tests\test_phase2_compressed_proto_knn.py` | Git-backed测试已创建 |

| 验证 | 结果 |
|---|---|
| RED | `conda run -n ssr-gpu python -m pytest -q code/tests/test_phase2_compressed_proto_knn.py`先因`ModuleNotFoundError: phase2_compressed_proto_knn_sweep`失败 |
| GREEN | 同一测试通过：`2 passed` |
| 语法 | `conda run -n ssr-gpu python -m py_compile code/scripts/phase2_compressed_proto_knn_sweep.py`通过 |

## 远端验证计划

| 项 | 值 |
|---|---|
| 远端脚本 | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_compressed_proto_knn_sweep.py` |
| 首轮目标 | `7-14`K=10近边界候选和`7-7/8-8`K=10新类强但旧类弱候选 |
| 扫描 | `prototypes_per_class in {1,2}`，`old_bias_grid=0..0.20`，`radius_weight_grid={0,0.25,0.5}` |
| 成功判据 | `old_acc>=0.85`、`seen_new_acc>=0.80`、`min_seen_new_class_acc>=0.85` |

## 当前状态

| 项 | 状态 |
|---|---|
| Git提交 | 待完成 |
| 远端同步 | 待完成 |
| 远端运行 | 待完成 |
| 结果解析 | 待完成 |
