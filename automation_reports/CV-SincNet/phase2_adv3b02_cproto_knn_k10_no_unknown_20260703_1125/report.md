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
| 最新补充口径 | 2026-07-03更新：以5/10-shot为标准，同时补充观察1-shot；5-shot性能相对1-shot可允许下降3pp，但主完成判据仍需同一行满足旧类>=85%、新类均值>=80%、每新类>=85%、新类数>=2 |

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
| 远端同步 | PASS：远端`py_compile`通过，脚本SHA256=`826a4c89f88c035b4ae5d664e94f4d4de51910b81b8173049bb86ff520fdebb3` |
| 远端运行 | 已完成：`7-14`、`7-7`、`8-8`三个K=10候选池均完成 |
| 结果解析 | 已完成：summary和日志已拉回 |

## 结果

| case | rows | 同一行达标数 | 最优行 | 解释 |
|---|---:|---:|---|---|
| `cproto_rx7_14_k10` | 810 | 0 | `19-3,6-6`，`cproto_p2_oldbias0_rad0`：旧类75.83%，新类均值60.00%，逐类最低57.50%，存储原型16，存储support0 | 压缩子原型不如原始KNN1，细粒度新类边界丢失明显 |
| `cproto_rx7_7_k10` | 810 | 0 | `14-13,15-19`，`cproto_p1_oldbias0_rad0.25`：旧类77.50%，新类均值80.00%，逐类最低77.50%，存储原型8，存储support0 | 不保存support成立，但旧类不足且逐新类不足 |
| `cproto_rx8_8_k10` | 810 | 0 | `18-17,2-7`，`cproto_p1_oldbias0.2_rad0.5`：旧类76.25%，新类均值78.75%，逐类最低77.50%，存储原型8，存储support0 | old_bias不足以把旧类推到85%，继续增大bias会损伤新类 |

## 产物与哈希

| 文件 | SHA256 |
|---|---|
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_cproto_knn_k10_no_unknown_20260703_1125\cproto_k10_summary.json` | `4820e15179c787846d850a4ed293d5c0698e68e2faf3e557874ba0a6d986d683` |
| `cproto_rx7_14_k10.out` | `a86516ef7432ee2ad5f0d7ef2382d1835b433e65558aa11cc2685c7d6c17f4bc` |
| `cproto_rx7_7_k10.out` | `278bc5e4bc861df5a8898ccc4a8a1fdef1c936993097aa900dbc0e0333999f59` |
| `cproto_rx8_8_k10.out` | `7136b89c8cbd86aaffb3aed8fad99cccce4f46413ca69754e49d7aac9a2fa278` |

## 方法结论

压缩原型KNN满足部署创新点：推理侧`stored_support_count=0`，只保存8到16个原型统计量。但当前单纯均值/子原型压缩损失了原始KNN1依赖的近邻细节，K=10未达到目标。后续如果继续沿此方向，应采用“可学习或选择式压缩”：每类保存少量代表性anchor/medoid或蒸馏原型，而不是简单均值；同时对旧类做目标域adapter或特征修复，再将修复后的support压缩进记忆库。

## medoid-anchor扩展计划

| 项 | 值 |
|---|---|
| 新增脚本能力 | `prototype_mode=medoid`：每类从support embedding中选择少量代表性anchor，先选最接近类均值的样本，再用farthest-first覆盖类内多样性 |
| 存储边界 | 不保存原始IQ，不保存全量support；只保存每类`M`个归一化anchor embedding、半径、计数和old/new标记。若K=10、8个类、`M=4`，存储32个anchor而非80个support embedding |
| 本地测试 | PASS：`conda run -n ssr-gpu python -m pytest -q code/tests/test_phase2_compressed_proto_knn.py`，3个测试通过 |
| 本地语法 | PASS：`conda run -n ssr-gpu python -m py_compile code/scripts/phase2_compressed_proto_knn_sweep.py` |
| 本地脚本SHA256 | `7C93E61B043B3C3673D71A0429714E615872875A85D2265C21C84D5E923BE531` |
| 待验证 | 在`7-14`近边界候选、`7-7/8-8`新类较强候选上扫描K=1/5/10、`M in {1,2,4,6,8}`、`old_bias`和`radius_weight`，记录同一行是否达标及5-shot相对1-shot差值 |
