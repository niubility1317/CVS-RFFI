# 源特征约束压力与有界度量检验

- run_id：`20261031-diagnostic-DURA-source-boundedTangent-s2026092701-r01`
- group_id：`phase1-dura-source-bounded-feature-metric`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

完整source63000既有F导数缓存谱压力，固定alpha谱裁剪使附加loss/IQ输出梯度不超过重建项；纯repeat固定512×两view验证特征方向保留；CPU2/0新G训练TX-V-target。

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


## 固定压力诊断与研究规则

只复用原F50/sourceclean的63000个固定J和原pure repeat E30/source512×缓存01/20误差，不加载producer G，不消费其target输出或TX真值。alpha保持0.0033130025035573727。以原IQ归一化能量E裁剪J的5个特征方向：lambda上限1/(alpha E)，弱方向保持，FP64分解后回到FP32参考缓存。附加loss及IQ输出梯度范数对任意误差不超过原IQ项；不保证G参数梯度、真实身份或目标收益。

先固定资格：全source至少5%实际需要cap、实际FP32因子的最大相对谱值≤1+2e-5；缓存01/20都要求旧实际guide/IQ梯度比p95>1、512cos可定义、新平均cos≥0.5且比旧最多低0.02、至少384条非负、每源RX均值非负、实际新梯度比max≤1+2e-5。失败不放宽，无自动训练。全63000谱压力/role-RX-day/trace集中度及双512数组保存，不以部分样本代替。

CPU2/CUDA空，新增F1024+首步2Ls、1024次输入VJP，0G/optimizer/TX/V/target。F状态保持，旧clean特征512引用按同head子空间变换。新因子缓存645.12MB只作源训练参考，不是新恢复网络或oracle部署。4聚焦检查和唯一审查后按一次最小发布流程执行；通过也需独立登记singlejoint30和全七景正式测试。默认保持，完整目标ACTIVE。


## 完整诊断结果

VERIFIED / ANALYZED / SOURCE_ONLY。全63000因子谱上界和2048条几何独立复算；旧实际压力p95=1.610377/1.349333，谱限制后平均cos.575031/.587405低于旧.643863/.661082超过.02，双源资格FAIL，不启动该G、不放宽规则。CPU7.343秒、源因子制备4.855秒/645.12MB，F1024+2smoke、G新前向/训练/TX-V-target0，F不变。见[完整分析](analysis.md)与results.json；默认保持/目标ACTIVE。
