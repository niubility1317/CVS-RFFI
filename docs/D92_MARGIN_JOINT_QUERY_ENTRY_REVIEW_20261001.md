# MarginJoint query 入口直接正确性审查

本次仅审查冻结源码、`configs/d92_margin_joint_repeat_20261001.json` 与该 run 的预登记文字。未发现会造成跑错 row、越权、覆盖、无法预测的 P0/P1。此结论不证明真实输入可用、远端发布成功或预测已经完成；本次未启动 query benchmark。

## 范围与证据边界

- 主入口：`tools/evaluate_d92_margin_joint_benchmark.py`、`tools/run_d92_margin_joint_benchmark.py`、`tools/publish_d92_margin_joint_benchmark.py`。
- 直接来源 helper：`export_d92_branch_features.load_features`、`export_d92_mv_kme_features.validate_origin/validate_capsule`、`d92_orbit_feature_cache.validate_split`。只读这些函数的源码。
- 预登记：`automation_reports/CV-SincNet/20261001-phase2-d92-margin-joint-repeat-m2-r01/experiment.json` 与 `report.md`。只读取文字与配置；没有打开其所引用的产物。
- 未读取真实 cache、packet、checkpoint、query、truth、预测、成绩、索引或交接；未执行数值计算、测试、Conda、Git、SSH 或发布。本次不重复审查 core 数学。
- root 已告知 24 项定点回归通过。该测试结论属于 root 提供的证据，本审查者未执行或重复测试。

## 已核对的启动与权限链路

| 检查项 | 源码位置与结论 |
| --- | --- |
| 四行与来源绑定 | `run_d92_margin_joint_benchmark.py:68` 至 `:115` 校验完整 cohort×模型集合、独占 row 输出和输入输出分离。同模型只固定 checkpoint SHA 与 Ground packet，允许跨 cohort 的 `row_root` 不同。当前 spec 明确声明 rx3/rx1×2026092701/2026092702 四行。 |
| 冻结资源与 CPU | Spec 明确 `max_transitions=4096`、`max_factor_buffer_bytes=167772160`（160 MiB）、CPU 两 lane、BLAS 两线程。`run_d92_margin_joint_benchmark.py:78`、`:272`、`:318` 强制两 lane/两线程环境和空 `CUDA_VISIBLE_DEVICES`。Predictor `:45`、`:215` 将冻结算法及两个显式 QP 限额逐次传给 prepare。此 factor 上限没有被描述成进程 RSS 上限。 |
| 合规只读输入 | Predictor `:63` 调用纯 cache 绑定 helper。`export_d92_branch_features.py:124` 至 `:171` 核对实际 capsule/checkpoint/model/source-only final200/类序/feature 契约、冻结访问声明和全部物理 ID，读取 received NPZ 时仅取 ID；不调用 encoder 或加载 checkpoint。Predictor `:71` 至 `:106` 将原 Ground packet 的 checkpoint、raw float32 `z_id` 契约、旧类集合及实际包字节绑定到本 row。两种 feature 契约按各自规范核对，没有强行要求两者全字典相等。 |
| 完整 capsule 遍历 | Predictor `:78` 至 `:95` 遍历 capsule 下全部 split 文件，要求数量等于 manifest 的 `split_count`、split ID 唯一且文件名绑定。`validate_split` 的 `:90` 至 `:106` 校验合法 support 标签、support/query 物理隔离和全类 K 计数，并拒绝 query truth/role/count 字段。没有评分选 split 或抽样路径。预登记声明 rx3 900、rx1 300 个 split，每个模型完整运行，共 2400 个 parent；实际 manifest 内容属于本次未读取的输入证据。 |
| 当次 B→C | Predictor `:186` 至 `:232` 每个 split 单独建立状态字典。B 的 prepare 只接收该 split 的旧 support；C 的 prepare 接收合法全 support，并显式传入该次实际 B 对象。`:227` 要求 C 的 prior ref 等于该 B 的 final ref。`:208` 至 `:209` 在 new0 时直接复用 B 对象，不再 prepare/fit C。没有跨 split、跨 row 或历史 adapted state 继承路径。 |
| 逐样本全列竞争 | Predictor `:235` 至 `:258` 逐物理 query 构造单样本输入；A 保留原 native 六类列和 float32，B 使用实际六旧类列，C 使用实际全部注册列。`:152` 至 `:156` 对完整有限分数统一 argmax，记录只有 split ID、opaque query ID、类列、分数和预测。C alias 同时写入同一条 C record，new0 不额外调用 C score。 |
| Truth-last | Predictor 不接收 truth 路径。Supervisor `:119` 至 `:136` 只传声明的预测输入；`:156` 至 `:224` 独立读回三流覆盖、类列、固定 argmax、C alias、当次 B/C ref 和完成档案。`:320` 至 `:327` 只有全部四行完成才写 `all_predictions_fixed=true`，始终记录 `truth_read=false/scorer_invoked=false`。评分是后续独立动作，publisher 不自动调用 scorer。 |
| 输出独占与失败保留 | Predictor `:58`、`:171` 使用不存在输出及独占创建；各预测流与标记采用独占写入。`:286` 至 `:304` 保留技术失败档案、失败工作审计或未知说明和已完成工作。Supervisor `:270`、`:293` 独占新 run/row，`:315` 至 `:327` 仅失败所属 row，保留健康 row 和 partial，不自动重试。 |
| 发布与唯一启动 | Publisher `:101` 至 `:113` 绑定已独立核实推送的实际 HEAD 与准备 parent，并要求运行源码未漂移。`:116` 至 `:150` 校验独占 release/run、精确白名单归档和隔离 import，只启动一个 supervisor。`:187` 至 `:198` 独立读回实际 runtime commit 与 startup spec，区分准备 parent 与实际 release。`:238` 至 `:241` 对不确定远端动作保存 UNKNOWN 并要求只读 reconcile，不自动再发布或再启动。 |
| 健康任务不干预 | 入口没有 stop、kill、低分停止、补跑或修改旧输出的动作。Publisher reconcile 中 `os.kill(pid,0)` 仅观察所属 supervisor PID。所有写入限定新 release/run 及本次本地发布目录。 |

## 未完成状态

P0/P1 清单为空。真实 source/cache/packet 是否存在且与声明匹配、实际发布 HEAD 和远端启动/完成证据均未由本审查读取，仍为 N/A。运行时由既有 preflight、实际预测输出和 root 的独立读回判定；本审查不新增审批、receipt、数据重验或额外 launch gate。

预登记将本次明确标注为已评分目标的透明重复基准，不是新的独立确认。三阶段准确率、遗忘、H 及新旧差距均尚未产生；本文件不作性能判断。
