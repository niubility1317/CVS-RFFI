# 已完成24个多解耦模型：完整测试结果

状态：ANALYZED / VERIFIED。6组×4seed，200epoch最终权重；每模型10视图，每视图168000个物理query。全部240份预测完成后独立truth-last评分，指标二次复算通过。

以下均为4seed均值，单位%。原剩余12个训练不因本次评分调整。

|场景|native|real_views|L|LT|L_EG|LT_EG|
|---|---:|---:|---:|---:|---:|---:|
|clean|80.908|80.848|81.185|80.992|80.938|81.069|
|practical_high|79.040|78.818|78.904|78.894|78.780|79.024|
|practical_mid|76.995|76.789|76.874|76.852|76.773|76.999|
|practical_low_suburban|74.825|74.616|74.727|74.704|74.614|74.833|
|practical_high_urban|76.036|75.854|75.929|75.926|75.810|76.051|
|practical_mid_urban|63.596|63.444|63.556|63.461|63.468|63.566|
|practical_low_urban|53.319|53.212|53.279|53.207|53.213|53.290|
|leo_clear_weak|65.227|64.864|65.075|64.927|64.997|64.840|
|leo_low_elev_weak|61.900|61.534|61.731|61.607|61.710|61.471|
|leo_rain_weak|62.085|61.644|61.935|61.734|61.875|61.638|

完整均值、标准差、Macro-F1、最差RX：results/summary.csv；逐seed/RX/TX：results/scores.csv；逐day：results/day_scores.csv；配对差：results/paired_results.json。

独立测试只读原scratch E200模型；来源和实际checkpoint检查见evidence/readback.json。未修改健康训练或原控制器，原完整矩阵计划保留；结果禁止回流调参、选模或重跑。

K、新类、适应前后及H为N/A，本次为Phase1零适配测试。ManySig是地面代理；LEO模拟不是在轨验证。

执行提交：3b0a5070519cae70a6f0ccbce68b2376f8aa8a09

## 结果解释

L在clean为81.1848%，相对native的80.9083%提升0.2765个百分点，配对4/4个seed为正；最差源外RX准确率均值从69.0469%到70.3167%。但L的三种LEO弱场景平均为62.9138%，低于基线63.0705%；六种practical平均为70.5449%，低于基线70.6351%。

LT相对L的clean下降0.1926个百分点，4/4个seed为负。EG路径未形成统一收益。当前六组中，所有增强/解耦方案的三种LEO弱场景均值均低于native，不能宣称多解耦已提高信道泛化能力。单L的clean小幅收益与最差RX改善成立，完整R分支不在本次完成快照内。

全部3360条overall/RX/TX指标和960条day指标已保存，240个overall记录均为168000条query；本地再次用混淆矩阵复算accuracy一致。
