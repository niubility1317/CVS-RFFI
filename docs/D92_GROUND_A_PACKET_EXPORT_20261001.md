# Ground A 原始分类头小包导出

本工具仅从 owner 明确提供、已经按项目原有契约核实的 source-only checkpoint 中提取原始 CosFace 分类头。它不重建或运行 encoder，不读源样本、源逐样本特征、support/query、checkpoint 评分、历史索引或结果，不训练、不选模。原 `GroundClassifierA`、当前实验方法和运行进程均不改变。

## 显式输入

`export_packet(checkpoint=..., binding=..., provenance=..., output=...)` 接受 checkpoint 路径、已绑定的 head JSON 元数据、既有 source-only provenance JSON，以及不存在的输出目录。CLI 对应 `--checkpoint --binding --provenance --output`。

`binding` 必須完整包含以下字段，不接受隐含默认值：

| 字段 | 口径 |
|---|---|
| `checkpoint_sha256` | 已经登记的原始 source-only checkpoint 身份；与 provenance、feature contract 和实际文件一致。 |
| `checkpoint_weight_key` | 精确为 `id_backbone.cls_head.head.weight`。 |
| `ordered_classes` | 6 个唯一物理类 ID，严格按原始 classifier weight 的行序。不能用排序后的集合、cache 列序或原型行序代替。 |
| `scale` | owner 从实际原始 factory/head 核实的正有限 s。没有 30 或其他默认值，不从名字推断。 |
| `norm_eps` | 显式为原 native CosFace 推理使用的 `1e-4`。 |
| `feature_contract` | `GroundFeatureContract` 的完整字典：同 checkpoint 身份、`cache_key=z_id`、`source_tensor=feat_joint`、`representation=raw`、`dtype=float32`、`feature_dim=160`。 |
| `logit_corrections` | 必须已核实为 `none`。未知或启用 Dual/logit correction 时，裸 CosFace 路线不适用。 |
| `class_row_order_source` | 实际原始 classifier 行映射的来源说明。 |
| `factory_scale_source` | 实际原始 factory/head.s 的来源说明。 |
| `corrections_source` | 实际 correction 开关结论的来源说明。 |

后三项是已有核实事实的来源描述，不是新的签名、receipt 或审批。空值、UNKNOWN 等占位声明失败。工具不能根据一段描述替 owner 证明真实 factory 或行映射；缺少实际事实时 owner 不能生成猜测绑定来绕过接口。

`provenance` 使用已有 `MATCHED_SOURCE_ONLY_SCRATCH` verdict、`target_access_before_freeze=false`、空 `checkpoint_inheritance`、匹配 checkpoint SHA256 和六类 source registry。原 source registry 只约束类集合，其顺序不替代显式 classifier 原行序。既有 model seed、checkpoint epoch、source role comparison 如存在则保留；不推断缺失 seed。来源核实状态未知、继承污染或 target 接触均拒绝。这里消费原有核实结论，不重新验证数据或建立新的来源证明链。

## checkpoint 读取与精确提取

工具先从同一个已打开文件句柄计算既有 SHA256 身份，再 seek 回文件头，使用 `torch.load(map_location='cpu', weights_only=False)`。该调用兼容项目 Torch 2.1 的 native checkpoint，其中可能包含 RNG 和其他 native Python 元数据；**仅适用于 owner 已经核实的本项目原始 checkpoint，不能用于任意或不可信来源**。反序列化不调用模型 factory，不实例化 encoder；工具仅访问 `payload['model']`，不检查 `stats` 等评分字段。

只支持 native `payload['model']` state mapping，不回退到 EMA、teacher、其他 `state_dict` 容器或 prototype。State key 只去掉至多一层开头的 `module.`；任何 key 的规范化碰撞都失败，即便两个 tensor 值相等也不静默覆盖。多层前缀、近似 suffix 或缺失确切原 head key 不会被猜测匹配。

原 weight 必须是有限、strided、float32 `[6,160]` tensor；不把 half/double 权重静默转成 float32。工具用真实 `GroundHeadMetadata` 与 `GroundClassifierA` 构造函数验证 ABI，但不调用推理。原始 float32 位通过 Torch uint8/list 转成 bytes，不依赖 NumPy↔Torch 共享数组 ABI。

## 小包与读回

成功目录只有三个文件：

| 文件 | 内容 |
|---|---|
| `head_weight.float32.bin` | 6×160 个连续 little-endian float32，精确数值分量 3,840 B。 |
| `metadata.json` | 原 ABI head/feature metadata、逐行 class mapping、三个绑定来源描述、既有 source-only verdict 子集，以及 native float32/eps/scale/tie-break 推理口径。 |
| `complete.json` | 状态 `GROUND_CLASSIFIER_A_PACKET_EXPORTED`、已有 checkpoint 身份、实际文件大小和完整本地小包字节数。 |

Schema 为 `d92_ground_classifier_a_packet_v1`。完整包字节包含 weight、metadata 和 completion 本身。Completion 的 JSON 字节数先在内存解析自引用长度至稳定，再独占写入；随后由实际文件 stat 求和核对，不能用 3,840 B 冒充完整包或真实网络传输字节。导出同时记录实际 checkpoint 文件大小、加载时间及 Torch 版本，但不将它们当作星载推理成本。

`load_packet(packet)` 独立读回这个小包：检查完成状态、文件集合和尺寸、源身份与 ABI、class mapping 和推理声明，再从 little-endian float32 bytes 构造冻结 `GroundClassifierA`。它不加载原 checkpoint 或 encoder。Export 自己也调用该读回接口，并逐字节比较写出的权重与提取的原权重。文件尺寸检查不是密码学封存或新的 hash 链。

Native 推理仍由原模块实现：float32 normalize、`eps=1e-4`、`F.linear` 后乘实际 s，面对全部已声明 ground 类，精确并列时按原 head 第一列。Packet 不含 target 标签、role、配额、原型或拟合状态。导出和读回均不代表实际 A 已评分；`actual_A_evaluated=false`。后续 A/B/C 的同 row、同物理旧 held ID 配对由 owner 的独立评估入口另行完成。

## 独占输出与失败

已有输出路径在任何 checkpoint 读取前拒绝。声明本身非法时先拒绝，不创建目录。开始 checkpoint 身份核对后，工具拥有新建的独占目录；任何加载、提取、写包或读回故障均保留已有文件，并写 `export_failed.json` 记录阶段与错误。失败后不删文件、不覆盖、不自动重试。

若最后读回失败，已写出的 completion 也保留作为故障证据；`export_failed.json` 优先阻止 loader 使用该目录。调用方不能仅凭 completion 文件存在声称成功，应以无失败标记且完整读回通过为准。新路径的重新尝试由 owner 明确处理，不能覆盖失败包。

## 合成验证状态

测试只生成 synthetic Torch checkpoint、tensor 和声明。Oracle 从本地原 `CosFaceHead` 源码 AST 提取无修改的 class，避免 encoder import。覆盖完整 export→load→native float32 prediction、非默认 scale、零/极小范数、原始行序与并列、module. 前缀碰撞、错误 dtype/shape、非法来源与 correction、checkpoint 身份不符、缺失真实绑定、重复输出、损坏包和读回失败保留。

本 worker 未运行数值测试、Conda、SSH、Git 或实际导出，也未读取真实权重、样本、cache 或评分。代码与合成测试完成后由 root 串行验证；真实 classifier 行序、factory s 及其显式绑定仍由 root 从原始训练来源核实。没有实际 A 结果或性能结论。

## Packet-only 顺序 runner 与发布入口

新增 `tools/run_d92_ground_a_packet_export.py`，CLI 为 `--spec --commit`。它只按 spec.rows 原顺序调用既有 `export_packet`，再独立 `load_packet` 核对产物；一条 CPU lane、两个 BLAS/Torch 线程，没有模型训练、encoder 构造或样本推理。运行目录必须不存在，每个 row 的 packet 输出严格为 run root 下同名目录。首个失败即停止，保留已完成 packet、失败证据和未启动 row，不自动重试。

Spec 的必填结构为 `run_id/spec_path/code/execution/rows`。`code` 明确 release cwd 与 Python 绝对路径；`execution` 明确 remote run root、唯一 launch owner `root`、`cpu_lanes=1` 和 `blas_threads_per_lane=2`。每个 row 明确 `row_id/checkpoint/source_contract_ref/expected_source_contract_ref/initialization_ref/selected_checkpoint_ref/training_startup_ref/binding/provenance/output_root`。Binding 和 provenance 沿用本文件上文的冻结 packet ABI；runner 另要求 `provenance.checkpoint_epoch` 类型严格为 int 且等于 200（拒绝缺失、bool、float 或字符串）。scale 和类序仍无代码默认值。Row 数与完成计数来自实际 spec，不在 runner 内隐藏额外 row。

加载 checkpoint 前核对五类既有来源元数据：

1. 实际 source contract：`native_role_comparison=EXACT_MATCH`、`checkpoint_init=scratch_only`、`target_access_before_freeze=false`；其 `role_ids/source_rxs/source_days/ratios/split_seed/num_classes` 与 owner 显式指定的原始契约完全一致。只比较已有角色与物理 ID，不加载或重验源数据。
2. `classes` 使用实际 native ctx 写入的 classifier 原行序，与 binding.ordered_classes 完全一致。expected 旧契约的 classes 不参与替换或推断。
3. Initialization：`scratch_only=true`、空 `checkpoint_sources`、`target_contact=false`、`target_training_contact=false`、`source_roles=EXACT_MATCH`、`ema_origin=this_run_student`。
4. Terminal status：只解析 `schema=phase1_terminal_status_v2`、`selection_source=training_final_only`、确切 `selected_checkpoint` 路径和 `selected_checkpoint_sha256`。与当前 row 原始 checkpoint 和 binding 身份匹配后才允许 exporter 加载。
5. `training_startup_ref` 必须等于该 checkpoint 原始源目录下的 `startup.json`，不能借用另一行的 startup；只解析其顶层 `selection`，必须等于 `final_epoch_200`。`training_final_only` 本身未声明 epoch，不能从 checkpoint 文件名或其他默认值推断 200。Spec 校验和来源文件核对分别强制 provenance 的精确整数 epoch，加载前同时核对 startup 的固定最终轮次声明。

元数据读取器按顶层白名单解析，terminal status 的其他评分结构只做词法跳过，不反序列化或使用。这个核对是当前输入与已核实 source-only 来源的一致性检查，不新增 approval、receipt、来源链或 IQ 重验。

Runner 保存实际 spec/PID/argv/commit/解释器/CPU/线程设置到 startup；逐步维护 state，输出详细 stdout 和 `training.log`（文件名沿用日志习惯，内容是 packet 导出事件），另存完整 events、紧凑 JSONL/CSV。每行显示实际 head scale、eps、原列序和 checkpoint 身份；记录实际 packet 文件字节、导出耗时及可测进程峰值 RSS。GPU、网络增量传输和不可测资源为 null。累计 packet 字节不含 supervisor 日志或网络传输。成功产生 complete，失败产生 failed 并保留全部已有文件。

新增 `tools/publish_d92_ground_a_packet_export.py`，CLI 为 `--spec`，显式恢复仍沿用旧 publisher 的 `--resume-staged-commit`。远端运行代码白名单只有四件：`cvsrffi/__init__.py`、`d92_ground_classifier_a.py`、小包 exporter 和新 packet runner。发布工具自身及既有 support publisher/runner 作为三个明确的本地传输依赖一并纳入 committed/pushed 检查和 archive；不复制整个 code 目录，不包含 Conditional/Affine 训练或 scorer。

发布前，在本机临时目录只复制上述四个远端运行文件，使用当前 Python 的 `-I -s` 新进程隔离导入，检查四个模块都来自该目录。这只验证确切源码白名单的本地导入完整性，不声称远端导出已完成。原有 transport 继续检查 committed/pushed archive、SCP、独占 release/run 和唯一 supervisor；只替换成 packet runner 并移除与 CPU packet 导出无关的 GPU0 占用检查。任何已存在目标 run/release 均拒绝，不修改健康 release。

原 exporter 的 31 项合成测试已由 root 串行通过，耗时 2.82 s，证据前缀为 `.codex_tmp/pytest_utf8_1790823054586420100`。新增 runner/publisher 测试包括两行完整原生小包、加载前来源与行序拒绝、评分字段跳过、独占输出、第二行失败保留第一行、最小白名单隔离导入与 mock 传输；本 worker 未执行这些新数值测试或任何 Git/SSH/实际发布。由 root 串行验证和唯一 launch，本文不声明真实导出已完成。

独立 P1 修订补齐了 final200 的显式绑定：合成回归新增 epoch 199/201、缺失、200.0、bool、字符串拒绝，以及原始 startup selection 缺失或不是 `final_epoch_200` 的加载前拒绝。此修订仅影响 runner 的输入一致性检查，未改 exporter 或 Ground A core，也未重验任何数据。
