# Phase1 CLIC七门终裁报告

日期：2026-08-16

状态：`FINAL / NOT_PROMOTED / PHASE1_GATE_FAIL`

## 结论

本轮Phase1候选不晋级。目标侧12臂预测和独立truth-side评分、source侧六fold指标、C控制描述器及G部署包均已完成技术闭合；但source非补偿门仅F1通过，真实未知显式拒识率全部远低于70％。ADV3B02合法盲预测与reference未能形成，因此target-known非劣关系只能记为`CANNOT_ESTABLISH`，不能写成通过或性能失败。

本目标到此停止，不再依据target结果调阈值、选fold、重训或追加ADV修复。Phase3仍未开始。

## 核心证据

- source：`phase1_clic_source_metrics_20260816_v4`，6fold×C/G同row clean、三种LEO weak及proxy指标已封存。
- target prediction：`phase1_clic_target_prediction_20260812_v1`，12/12预测，每臂3120行、3120次forward，truth未打开，fit/update/retry/selection均为0。
- target metrics：`phase1_clic_target_metrics_20260812_v1`，12/12独立评分receipt，known、unknown、open-set、FR/defer/coverage及scene/RX/class/day DG完整。
- deployment：C控制描述器6/6可用；G safe-pack bundle 6/6完成production verify/reload。
- ADV3B02：v1—v4均只到技术smoke且停止，正式盲预测调用为0，未生成合法reference。

## 七门终裁

|门|终裁|核心证据|
|---|---|---|
|1.训练与技术工件闭合|PASS|12/12 checkpoint与terminal闭合；C描述器6/6、G bundle 6/6；source与target正式工件完整。|
|2.source-known clean与三种LEO weak|FAIL|六fold aggregate为false；仅F1 fold verdict=true，F2—F6=false。|
|3.source-proxy正向信号|FAIL|仅F1 proxy gate=true；F2—F6=false，不能由其他fold补偿。|
|4.target-known相对ADV3B02非劣|CANNOT_ESTABLISH|合法ADV盲预测、rich reference和combined score均未生成；formal=0。|
|5.真实unknown显式拒识≥70％|FAIL|12/12均失败；C全局拒识率1.92％—8.00％，G为1.29％—6.12％，所有scene均低于70％。|
|6.known FR、unknown FAR、coverage、defer及DG封存|PASS|12/12 target metrics receipt完整，且每项来自同一候选同一行。|
|7.真实checkpoint部署工件|G PASS；C为control artifact|G bundle 6/6 verify/reload通过；C 6/6描述器已实际完成预测，但不冒充G式新方法bundle。|

七门不可补偿。门2、门3和门5的失败已足以否决晋级；门4缺失也不能由其他门替代。

## source同row门结果

|fold|floor gate|proxy gate|scene-equal gate|fold verdict|
|---:|:---:|:---:|:---:|:---:|
|F1|true|true|true|true|
|F2|true|false|true|false|
|F3|false|false|true|false|
|F4|false|false|true|false|
|F5|false|false|true|false|
|F6|false|false|true|false|

六fold aggregate=`false`。18个scene的equal-overall汇总门为true，但不能补偿floor或proxy失败。

## target同row核心结果

|fold|known overall C/G|unknown显式拒识 C/G|70％门|
|---:|:---:|:---:|:---:|
|F1|67.29％／65.21％|3.87％／6.12％|FAIL／FAIL|
|F2|61.04％／61.67％|8.00％／2.71％|FAIL／FAIL|
|F3|71.88％／71.88％|1.92％／2.58％|FAIL／FAIL|
|F4|71.46％／69.17％|2.71％／1.29％|FAIL／FAIL|
|F5|42.29％／44.58％|1.96％／2.71％|FAIL／FAIL|
|F6|48.33％／50.21％|2.67％／2.67％|FAIL／FAIL|

known结果仅用于描述已封存表现；由于ADV reference不存在，本表不产生非劣结论。unknown结果是有效性能失败，不是技术失败。

## 最终决定

- 当前候选：`NOT_PROMOTED / PHASE1_GATE_FAIL`。
- ADV3B02比较：`CANNOT_ESTABLISH`，不是baseline性能失败。
- 本目标内不再重试、调参或扩展证明层。
- 若未来提出新候选，必须作为新的source-only冻结实验重新开始；不得利用本次target结果做候选选择。
- Phase3多接收节点代理协同、unknown生命周期与注册授权继续保持`PENDING`。
