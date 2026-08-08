# Phase1 DualReadout-Disagreement窄实验设计

状态：`DESIGN_FROZEN`

目标模式：`GOAL_MODE=ACTIVE`

## 1.首轮证据

GeoSat Lite四臂证明：B的known-only角度几何相对A把proxy FAR从53.50%降到38.25%，但没有LEO增益；C的clean→LEO单向KL把LEO mean/floor提升9.067pp/9.387pp，但proxy FAR升到66.00%；D把两项压入同一路径后proxy FAR进一步恶化到79.25%。因此下一步不再增加对齐或共同loss，而是解耦读出职责。

## 2.冻结方法

- angular readout：冻结B checkpoint，只产生source-calibrated confidence、margin、energy和类别预测；
- robust readout：冻结C checkpoint，只产生最终registered class预测；
- disagreement：同一物理样本上计算两组softmax的Jensen-Shannon divergence；
- 配对：必须逐行匹配非空`sig_id`以及TX/RX/day/equalization/view元数据；缺ID或同桶行重排直接失败；
- 校准集合：仅source role且B、C均正确的样本；冻结`JS Q0.95`；
- 接受规则：B的confidence/margin/energy三门均通过、B/C top-1一致、JS不超过source Q0.95；
- held-known和proxy-unknown不参与阈值、公式或模型选择；不训练、不扫参、不fallback。

该方法输出“C类别证据+B拒识证据+跨目标分歧”，不是logit平均或多数投票。它仍是Phase1 source-only开发拒识，不是Phase3真实unknown。

## 3.冻结矩阵

只运行一条方法、两条同公式诊断：

| row | known | unknown | 目的 |
|---|---|---|---|
| proxy | source四TX | proxy TX `8-20` | 主要source proxy信号 |
| held-known | source四TX | held TX `6-15` | 第二未见TX方向一致性 |

输入直接复用`postfreeze_audit_v1`中B、C的同序NPZ，不重新过backbone。固定quantile：confidence=0.05、margin=0.05、energy=0.95、JS=0.95；目标FAR=0.05。

## 4.判定

本轮只问解耦是否比单B更有信号：proxy FAR必须低于38.25%，source full accuracy相对B不得再下降超过2pp，held-known FAR不得劣于B的21.25%。任一不满足即拒绝该组合；即使满足但FAR仍高于5%，也只能进入下一次LEO弱信道窄实验，不能导出正式deployment bundle。
