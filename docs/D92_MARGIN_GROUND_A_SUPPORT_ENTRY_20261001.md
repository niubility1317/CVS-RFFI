# MarginJoint 的 Ground A 支持集配对入口

本次仅扩展已有 Ground A scorer、supervisor 和报告工具，使其接受冻结 MarginJoint 的实际固定预测。不改变 A 分类头、Margin 训练、方法参数、QP 资源、任务矩阵或已有分析结果。不创建真实配置，不发布或启动实验；root 是后续唯一运行与验证 owner。

## 明确的方法与来源契约

以下身份直接对应冻结的 `tools/evaluate_d92_margin_joint_probe.py`：

| 字段 | 值 |
|---|---|
| method schema | `d92_margin_joint_local_ridge_v1` |
| method | `D92-MarginJointLocalRidge-v1` |
| lane completion | `MARGIN_JOINT_PROBE_COMPLETE` |
| scope | `SUPPORT_ONLY_MARGIN_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION` |
| fixed candidate path | `R_MARGIN_seq` |
| actual B / C stage | `B_MARGIN` / `C_MARGIN_seq` |
| independent summary status | `COMPLETE_MARGIN_JOINT_PROBE_VERIFIED` |

supervisor 的 spec/API 外形不变。它仅支持两种明确组合：

1. 原 Affine＋Conditional 八行：每种方法来自一个独立 source run，各四行。
2. 新 Margin 四行：全部来自同一个 Margin source run，两个显式 model seed 与两个显式 capsule 的完整 2×2 交叉。每个 seed 的两行必须指向同一 checkpoint SHA256 和同一原 Ground A packet；两个模型明确对应两个不同 checkpoint。

不接受任意方法组合、混合 Margin 与旧方法、重复 source row、拆分 source run 或缺失模型/cohort 组合。每行仍显式提供 source run/row、packet、support cache、fit trace、source summary、预期 status/schema/method/runtime commit、checkpoint SHA256/capsule/model seed 和新独占 output_root。

来源检查仍只白名单反序列化 summary 与 lane startup/complete 元数据。每个 source run 必须有完整 160 parents / 1800 paths；四个声明 lane 总数与该证据一致，query/source 使用量均为 0，旧类数为 6。完整 source selection 与当次 cache/五项身份绑定保持。不会为了补 A 读取原结果表来选择配置或参数。

## 固定预测后的配对

scorer 的 Python 接口不变：

```text
score_support(packet=..., support_features=..., fit_trace=..., output=...,
              run_id=<source-run>, row_id=<source-row>, selection=None,
              expected_checkpoint_sha256=..., expected_capsule_id=...,
              expected_model_seed=...)
```

先按显式当前 row 的 support selection 读取 raw float32 z_id，原 packet 的六个 native class columns 对每个物理记录统一竞争。A 全部 scores/predictions 持久化并读回后，才连接合法 support truth 和已有 B/C fit trace。没有 encoder 运行、校准、prototype 替代头或适应操作。独立 truth join 不再调用模型。

Margin entry 必须具有 `FIXED_BEFORE_SUPPORT_TRUTH_JOIN`（有 held）或 `NO_HELD_PREDICTIONS`（真实 K1）。旧/新训练与 held IDs、全注册列顺序、同 run/row/split/fold/trial 归档绑定继续逐路径检查。新类存在时，还要求 C preparation 的继承标记、训练 IDs、`preparation_ref` 和 `inherited_from_B` 一致，且 C final problem 的 `prior_ref` 精确等于同路径实际 B 的 `final_state_ref`；嵌入 C stage 的 preparation 也须相同。不接受 B0、另一个 B 或其他路径替代。

new0 只有一个实际 B stage/preparation，C 复用该 B 的固定 scores。真实 K1 没有独立 held A/B/C 或 B−A，仍为 N/A。Ground A 缺失时不能用 R0、B0 或缺省值替代。固定 A 已输出而配对失败时，保留 A 及失败证据，不覆盖原文件、不重试。

## 报告矩阵与解释

报告继续要求每行 40 parents、其中 K1 为 10 parents，以及每方法四行共 160 parents。每方法 `by_k_new_count` 有 **40 个 cells**：两个诊断（OOF、proxy）×四个 K（1/5/10/20）×五个新增类数（0/2/5/10/20）；原两方法合计 80 cells，新单 Margin 合计 40 cells。没有虚增单方法矩阵。

每个 cell 必须含全部八项指标；真实 K1 和 new0 的缺失规则不变。旧类 A/B/C 来自同一物理 held support，report 再核对 source 五项身份及 Margin model-row 表的 seed/row 身份。H、B−A、注册后旧类下降与绝对新旧差保持 scorer 的 parent-first 计算，不从汇总准确率重算。proxy 先在每个 parent 内平均全部 anchor，再平均 parents。缺 A、缺矩阵 cell、错方法或错物理绑定都不能形成完整报告。

报告资源表引用当次 pairing.resources 的实测值；不再复制历史 packet 字节常数。文件或 numeric array 字节不代表星地传输、设备常驻内存或能耗，未测项保留 N/A/null。理想改善目标继续是 soft 目标，不新增性能硬门槛。

这些结果仅为合法 support held 诊断，不是 query/generalization 结论，也不是额外独立数据验证。补充结果不能回流适应、选模或选择性重跑，原训练 trace 和独立 summary 保持只读。

## CLI 与验证范围

原命令行保持：

```text
python tools/run_d92_ground_a_support.py --spec <owner-spec> --commit <actual-runtime-commit>
python tools/score_d92_ground_a_support.py --packet <packet> --support-features <cache> --fit-trace <trace> --output <new-output> --run-id <source-run> --row-id <source-row> --expected-checkpoint-sha256 <sha256> --expected-capsule-id <capsule> --expected-model-seed <seed>
python tools/report_d92_ground_a_support.py --spec <owner-spec> --raw-root <supplement-run> --report <new-report.md> --interpretation <new-interpretation.json>
```

本次修改三个工具、各自测试和本文档。publisher、core、训练/分析入口和配置均未修改；发布白名单继续由 root 整合。不增加新的运行依赖，原 scorer 测试仍需要 Torch 与 NumPy；报告本身仍仅依赖 stdlib。

root 串行验证命令：

```text
python -s -m pytest tests/test_score_d92_ground_a_support.py tests/test_run_d92_ground_a_support.py tests/test_report_d92_ground_a_support.py
```

测试保留原八行契约，新增 Margin 生产合成 trace 的固定分数/真实 B→C 引用、new0、K1、truth-last、缺 A 和篡改拒绝，四行显式 2×2 来源、完整矩阵、parent-first 指标及实测资源引用。开发阶段仅进行 UTF-8/AST 静态检查，未运行数字测试、Conda、Git、SSH，未读取真实 packet/cache/trace/结果。

root在verified ssr-gpu环境串行执行三文件：150 passed、12 skipped、1 known Torch/NumPy ABI warning，15.73s。12个skip为两种旧方法fixture上不适用的Margin专用断言，Margin production trace与配对测试已实际执行。证据pytest_utf8_1790841033042302100。未做真实Margin评分或模型微调。
