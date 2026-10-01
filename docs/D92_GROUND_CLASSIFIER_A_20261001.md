# Ground-A 原始分类头推断接口

日期：2026-10-01。状态：`SYNTHETIC_VERIFIED / ACTUAL_PACKAGE_NOT_INTEGRATED`。

[实现](../code/cvsrffi/d92_ground_classifier_a.py)只执行冻结 ground head 的无标签推断。它不读取包、checkpoint、cache 或数据文件，不拟合 target support，不使用标签、role、配额或全局排序，也不以 prototype、R0 或 B0 替代 A。实际权重、s、head 列序及 Dual corrections 尚未完整绑定；本实现不产生 `actual_A` 或真实来源已验证的状态。

## 输入与输出

调用方必须显式构造以下三个对象；实际 s、类序和 eps 均无默认值。

| 对象 | 必填内容与边界 |
|---|---|
| `GroundFeatureContract` | 原 checkpoint SHA256 身份、`cache_key='z_id'`、`source_tensor='feat_joint'`、`representation='raw'`、`dtype='float32'`、`feature_dim=160`。此处使用已有 checkpoint 身份，不新增文件哈希或证明链。 |
| `GroundHeadMetadata` | 同一 checkpoint 身份、原 weight key `id_backbone.cls_head.head.weight`、六个唯一且按原头排列的物理类 ID、实际正有限 scale、实际 `norm_eps=1e-4`、对应 feature contract，以及已记录的 `MATCHED_SOURCE_ONLY_SCRATCH` 来源结论、`target_access_before_freeze=False`、空继承列表和明确的 `logit_corrections='none'`。这些是调用方的冻结声明，不是本模块独立核实的来源事实。 |
| `GroundClassifierA` | 原始 float32 `[6,160]` weight tensor 和上述 metadata。构造时复制准确 float32 位表示到不可变 bytes，外部 weight tensor 后续修改不会改变 A；metadata 转为冻结对象并保留原始类序。 |

`score(z_id=..., feature_contract=...)` 只接收 torch float32 `[N,160]` 原始 `feat_joint`，要求 feature contract 与冻结 head 完全一致。本次项目环境的 Torch 2.1.0 与 NumPy 2.2.6 存在 C-ABI 警告，不能假定 `torch.from_numpy` 可用。未来 I/O 层可沿用现有冻结分支导出的 scalar-list 桥接，显式用 `torch.tensor(z_id.tolist(), dtype=torch.float32)` 保留原 float32 数值；桥接耗时另测。本模块不接受 float64 输入并静默转换，也不接受已归一化/换名称/其他 checkpoint 的 feature contract。空输入输出 `[0,6]`；非有限值失败。

计算严格沿用 [原 `CosFaceHead`](../code/model.py) 的无 labels 分支：在禁用当前设备 autocast 的上下文中，用 float32 执行 `F.normalize(z_id.float(), dim=1, eps=1e-4)`、同样归一化 weight、`F.linear`，最后乘实际 s。没有标签 margin、softmax、温度、prototype 或 float64 矩阵替代。仅支持 CPU/CUDA；权重按输入设备物化，返回 float32 scores，不保留梯度图。

`classes` 原样返回冻结列序。`predict` 返回原列序 `argmax` 对应的类 ID，精确并列时选择原头第一列，不改为字典序。每个记录面对全部已声明 ground 类；不同 batch 切分可能产生底层 float32 GEMM 的舍入差异，但没有跨记录统计、配额、query 图或重排。调用方保留 scores 和列序，不能从该 API 推断尚未声明的新类。

## 包导出与 A/B/C 配对的接入边界

实际包导出由主任务的 I/O 层完成。本模块不选 checkpoint，也不猜测 s、类序、weight 或可选 Dual 修正。导出层需从实际冻结 source-only checkpoint/config/class mapping 取得原权重、实际 s/eps 和有序物理类 ID，并声明 raw cache 的同 checkpoint `feat_joint` 契约。若 Dual 或其他 logit correction 状态未知或启用，当前裸 CosFace 路线拒绝接入；不能把未知写成 `none`。若原头不是六类或输入不是 raw 160 维，本最小接口不适用。

最小调用关系是 `GroundClassifierA(weight=actual_weight, metadata=actual_metadata)`，随后传入合法物理记录的 raw float32 特征及已绑定的 `GroundFeatureContract`。文件格式、读取路径和实际来源核验仍属于已有 I/O/provenance 流程；这里没有新包格式或 receipt/signature/authority 要求。

A/B/C 配对由外部评估层依据同一 row、同一物理旧类 held IDs 完成：A 用冻结 ground head，B 用该 row 的旧类 support 适应状态，C 用加入新类后面对全部注册类的状态。A 的六列不能冒充 C 的全部注册类；原 ground 类序与 B/C 列序必须通过物理类 ID 显式对应。共同样本上的预测固定后再连接 truth 计算各阶段旧类指标及差值。本模块不接收 held truth，也不负责选择 held IDs。缺少实际 head 或配对证据时 A 与 B−A 保持 N/A，不能补用 B0。

## 合成验证与限制

[合成测试](../tests/test_d92_ground_classifier_a.py)从本地 `code/model.py` 的 AST 提取未修改的 `CosFaceHead` 类作为 oracle，避免实例化 encoder 或读取真实权重。测试覆盖 float32 scores、实际 scale、零/极小范数及非单位特征、原类序与并列、列置换、逐样本/空输入、autocast 禁用、无梯度、冻结权重和元数据、非法契约、未知来源/corrections、不同 checkpoint、非法 tensor，以及标签/role/配额/拟合相关输入拒绝。

子任务未运行测试、Conda、SSH、Git 或实验，未读取真实权重、cache、评分、索引或交接。主任务在工作区 `local_envs/ssr-gpu` 补齐官方 CPU-only Torch 2.1.0 后，串行运行本文件测试：**43 passed, 1 warning in 2.46s**。测试与本地原 `CosFaceHead` 的同设备 float32 scores 位级一致；不代表跨设备或真实导出包已验证。警告为上述 NumPy C-ABI 不兼容，测试没有调用 NumPy↔Torch 共享数组接口。完整输出为 `E:/type10-7/.codex_tmp/pytest_utf8_1790816994220795100.stdout` 与同前缀 `.stderr`。[官方固定版本 CPU 安装说明](https://pytorch.org/get-started/previous-versions/#v210)仅用于本地合成测试依赖，不是星地传输数据。

主任务另以只读方式核实两个实际 selected source-only seed 的 `source_contract.json`：classes 均为 `14-10,14-7,20-15,20-19,6-15,8-20`，初始化均为 scratch-only、无上游 checkpoint、无 target contact，选模为 final epoch 200。实际 `resolved_config/startup.resolved_args` 均为 `representation_mode='dual'`、`sat_anchor_adapter=false`、`id_feature_key='feat_joint'`，没有上述两类 Dual correction。这补齐了 source contract 的实际类列表和实际 correction 开关，尚未代替原 head 行与 loader 列序的绑定。实际 factory 模块及 `head.s` 仍待核实。只读证据未加载模型 tensor、源样本或 target query；没有接入 A 评分，A、B−A 继续为 N/A。

本模块不测量完整部署包、星载耗时或内存峰值，不据此声称省算力。原 head 的 float32 权重形状是 `[6,160]`，其数值分量大小为 3,840 B；实际包/增量传输字节仍为 N/A，不能把这一形状计算冒充实际传输测量。
