# 新整体PAMF仅去除SIG-TIME：纯CE E200八种子匹配消融

- run_id：`20261026-phase1-pamf-nosigtime-purece-e200-manysig-m8-r01`
- group_id：`phase1-pamf-isolated-sigtime-ablation`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Scratch matched whole-network ablation: ReferenceCVS/NoExplicitCFO/compensation/default head retained, only SIG-TIME sequence path removed.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 固定实现和验证

只去除SIG-TIME的6维投影、四窗路径增量/反对称面积与残差融合。ReferenceCVS参考响应、NoExplicitCFO（坐标27常数零）、内部补偿及原默认头保持。完整模型196001参数，消融177025参数。共享参数初始化逐项完全一致，移除后的features精确等于完整模型的ReferenceCVS基线特征。两项聚焦测试（含5种非法训练配置）通过；原训练器与严格预测检查复用。

本次只测clean；同时保留全168000与D0+未见RX42000子集。已有12种常见网络和完整SIG_TIME不重训、不重新预测。全部报告按实际8seed均值/样本SD，退化种子不剔除。新整体模型指此历史纯CE SIG_TIME，不混入后来T1/EMA/X3训练配方。唯一P0/P1审查结果及真实CP smoke随发布记录。

唯一独立P0/P1审查PASS。D0索引和实际日期已从原ManySig metadata核实；原104历史预测与新整体8预测均存在。新主表12常见网络+SIG_TIME完整网络+严格无SIG_TIME，共14×8=112行；先生成已有13×8的双跨结果，8行消融完整后补齐，未完成不冒充112行。原residual_fusion与native不作为当前PAMF主表行。

发布前资源调整：GPU0/2/3/4已被其他健康任务使用。首次预检在任何SCP/远端写入前拒绝，尚未启动本run。实际改用空闲GPU5/6/7，一卡一条本run worker，8seed队列，科学配方不变。

## 最终结果与交付（2026-10-26）

VERIFIED。8/8严格无SIG-TIME消融从零E200、10000更新，全部完成clean168000预测与独立truth-last评分。全部200轮源日志复核纯CE/无U/无增强/无额外损失，默认头联合训练。

12常见网络＋完整SIG_TIME PAMF＋严格去SIG-TIME，共14网络×8seed。复用旧固定预测；D0=2021-03-01且RX未见的42000包是原168000包子集；原范围不含源RX的单独跨日测试。D0早于源D1–D3，不声称未来日期外推。原目标基准已暴露，属于事后分层统计，不反馈训练或选择。

|范围|网络|n|Accuracy（%）|Macro-F1（%）|
|---|---|---:|---:|---:|
|双跨|PAMF|8|79.6256 ± 1.6096|78.7900 ± 1.7198|
|双跨|PAMF w/o SIG-TIME|8|76.9801 ± 1.6892|76.5275 ± 1.8394|
|双跨|Complex ResNet-18-1D|8|75.0533 ± 1.3353|74.6846 ± 1.4902|
|双跨|ResNet-50-1D|8|74.7982 ± 1.6419|74.4013 ± 1.5546|
|双跨|ResNet-34-1D|8|74.3247 ± 2.4851|73.6110 ± 2.5327|
|双跨|DenseNet-121-1D|8|73.8738 ± 2.2741|72.9196 ± 2.3184|
|双跨|ResNet-18-1D|8|73.2932 ± 1.8433|72.9152 ± 1.7706|
|双跨|VGG-16-BN-1D|8|72.1685 ± 2.8374|72.2246 ± 2.4024|
|双跨|BiLSTM (3 layers)|8|69.2890 ± 1.3251|69.4621 ± 1.1448|
|双跨|ConvNeXt-Tiny-1D|8|65.4753 ± 1.0069|64.6618 ± 1.4957|
|双跨|ResNet-18-2D|8|65.3366 ± 3.9808|64.5924 ± 4.4109|
|双跨|ResNet-50-2D|8|64.5985 ± 1.8623|63.5314 ± 2.3092|
|双跨|Complex ResNet-18-2D|8|64.2759 ± 3.4621|63.9643 ± 3.9867|
|双跨|Transformer (6 layers)|8|53.4979 ± 22.8397|50.4498 ± 28.3225|
|综合|PAMF|8|81.5965 ± 0.3997|81.1798 ± 0.4680|
|综合|PAMF w/o SIG-TIME|8|80.3821 ± 1.2109|80.1351 ± 1.2484|
|综合|ResNet-50-1D|8|74.9862 ± 1.0593|74.7782 ± 0.8817|
|综合|DenseNet-121-1D|8|73.0940 ± 1.7436|72.3632 ± 1.9779|
|综合|Complex ResNet-18-1D|8|72.9711 ± 1.9541|73.0264 ± 1.8260|
|综合|ResNet-18-1D|8|72.7422 ± 1.2680|72.6778 ± 1.3360|
|综合|ResNet-34-1D|8|72.5284 ± 2.0459|72.3455 ± 2.2557|
|综合|VGG-16-BN-1D|8|70.1266 ± 2.0649|70.6634 ± 1.5015|
|综合|BiLSTM (3 layers)|8|67.3809 ± 1.1405|66.9636 ± 1.3252|
|综合|ConvNeXt-Tiny-1D|8|65.0792 ± 0.8017|63.9813 ± 0.9635|
|综合|ResNet-50-2D|8|64.8656 ± 2.1464|64.1647 ± 2.4495|
|综合|ResNet-18-2D|8|64.8075 ± 2.8478|64.5200 ± 3.2537|
|综合|Complex ResNet-18-2D|8|64.4991 ± 3.1440|64.3351 ± 4.0303|
|综合|Transformer (6 layers)|8|52.8267 ± 22.4267|49.7650 ± 27.8795|

严格消融配对差（完整减去除版）：

- 双跨：+2.6455 ± 1.8404个百分点；正差8/8，所有正负差保留。
- 综合：+1.2144 ± 1.3822个百分点；正差6/8，所有正负差保留。

完整网络196001参数，去除版177025参数。实际执行提交419663af5c2ccf27c7f5a1e6e4f8359388b3e39f；GPU5/6/7健康队列。

交付Excel：E:\type10-7\outputs\01a0fd26-2621-7662-a9bd-9ff19509e812\standard_ce_crossday_crossrx_pamf_20261026.xlsx；图包：E:\type10-7\outputs\01a0fd26-2621-7662-a9bd-9ff19509e812\standard_ce_crossday_crossrx_pamf_20261026_figures及同名ZIP。9个sheet、19张图及19页PDF、2张可编辑图表，所有汇总公式与评分一致，原Excel字节保持。完整独立验证：local_artifacts/pamf_nosigtime_purece_20261026_r01/delivery_verification.json。

最终视觉核验：仅将Fig05统计文字移到绘图区上方，避免与种子数据点相交；PDF/PNG/SVG、Excel嵌入图片与ZIP同步。其余18张PNG、所有Excel数据/公式/格式与对象数量均保持一致；核验见local_artifacts/pamf_nosigtime_purece_20261026_r01/figure_qa_verification.json。
