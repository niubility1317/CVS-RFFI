# Phase1 CLIC target LEO-weak 12臂指标封存v1报告

## 状态与目标

- 实验ID：`phase1_clic_target_metrics_20260812_v1`。
- 当前状态：`LOCAL_VERIFYING / FORMAL_LAUNCH=0 / RETRY=NO / NO_PERFORMANCE_RESULT`。
- 操作者：主控Codex；N607唯一runner：`Luna/max`。
- 目标：对已完成且truth-blind封存的`phase1_clic_target_prediction_20260812_v1`共12份prediction进行一次独立truth-side计量，输出target LEO-weak known、unknown rejection、open-set和scene／RX／class／day DG完整证据。
- 本run不训练、不适配、不更新、不重跑预测、不改变阈值、不选择候选，也不要求与后续ADV3B02共用同一物理包；只消费既有prediction和同一truth sidecar。

## 冻结输入与输出

- 输入prediction：`runs/phase1_clic_target_prediction_20260812_v1/predictions/F{1..6}{C,G}_CLIC12.prediction.json`共12份；上游报告commit=`08326e2a`，每份3120行、forward_count=3120、同一package SHA=`64373f39e45cb7dcfc5b5d0a989e43ba17f3ff16d85577631434e65e6235fff5`，truth未打开且fit/update/retry/selection全0。
- truth sidecar：`runs/phase1_clic_target_prediction_20260812_v1/sealed_target/truth_sidecar.json`；只能由本独立scorer在prediction/predictor/package验签后首次打开。
- 输出：`runs/phase1_clic_target_metrics_20260812_v1/metrics/F{1..6}{C,G}_CLIC12.metrics.json`共12份；日志：`logs/phase1_clic_target_metrics_20260812_v1`；两根启动前必须不存在。
- launcher：`code/scripts/launch_phase1_clic_target_metrics12_v1_20260812.sh`；N607`CVS-RFFI`Python，CPU执行，CUDA禁用，OMP／MKL／OpenBLAS各2线程。
- evaluator commit=`17e46320`，SHA-256=`845658D432891314447EFE171E91EC772181366F9A2B317595CE53EDFACA8052`；launcher SHA-256=`23BB602F5A026CC17E5886E4A6F7B52ED9756C56C0CA57463FDBECB6771F7A63`。launcher专测`2／2`、`bash -n`和dry-run12行（C6/G6）通过，禁ADV／combined score／prediction重跑／package／threshold／class-order／retry参数为0。

## 指标与证据边界

- known：registered-known中的unknown／defer均按错误计；按scene、receiver、class、day及三scene等权、sample-pooled封存准确率、macro、minimum、false reject、defer和accepted-known coverage。
- unknown：显式拒识只计`decision=unknown`，defer单列且不计分子；全局和每scene冻结分母、分子、defer和拒识率，预注册floor为0.70。低于floor是有效失败结果，仍写receipt而不抛异常。
- open-set：封存AUROC-unknown、AUPR-out、FPR95；prediction的`e_unknown`和decision只在truth连接后用于计量，绝不回流。
- 每份receipt的`passed`只表示unknown floor，不表示对ADV3B02非劣或方法综合晋级；固定`baseline_compared=false`、`comparison_status=ADV_COMPARISON_PENDING`，不得写ADV通过结论。
- 12份全部闭合后才读取性能并形成同一candidate/run行表。每行必须同时呈现target LEO-weak、unknown rejection、open-set、known/DG、defer/coverage和技术状态；不得拼接不同臂的单项极值。

## 本地验证与正式停止规则

- target-only sealer API／CLI、正例公式、69 unknown＋31 defer gate-false、prediction／truth／predictor篡改、truth打开顺序、真实G文件调用和不可覆盖测试`7／7`通过；完整postfreeze`152／152`通过；`py_compile`、`git diff --check`通过。
- 原combined ADV scorer保持严格，不接受缺失reference；当前无配置等价ADV原件，所以本run只封target metrics，后续另行生成6fold ADV baseline并完成非劣比较。
- formal launcher唯一调用1次，retry=`NO`。发生协议泄漏、hash/checkout漂移、覆盖风险或至少2fold同一确定性异常时，只停止本run确切进程并保留证据；不得按accuracy、unknown rejection或其他性能值停止。

## 结果表（完成后回填）

|候选|机制|target RX/day|LEO-weak场景|known overall/macro/min|unknown reject（全局/逐scene）|AUROC/AUPR/FPR95|defer/coverage|scene/RX/class/day DG|ADV比较|技术／最终结论|
|---|---|---|---|---|---|---|---|---|---|---|
|F1C—F6C|C：raw phase control|20-1；day0,1,2|clear/low-elev/rain|待12份receipt|待12份receipt|待12份receipt|待12份receipt|待12份receipt|PENDING|待运行|
|F1G—F6G|G：complex local invariant curvature|同上|同上|待12份receipt|待12份receipt|待12份receipt|待12份receipt|待12份receipt|PENDING|待运行|

本表当前仅预注册字段，不含性能数值；任何结果必须来自本run同一行receipt并保留candidate ID。
