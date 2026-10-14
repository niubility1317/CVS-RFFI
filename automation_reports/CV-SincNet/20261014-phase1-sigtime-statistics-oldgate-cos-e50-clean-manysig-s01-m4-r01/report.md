# SIG_TIME 综合局部均值方差与协方差：固定首种子实验

用户明确要求将并列 MEAN_VAR/COV 综合到 SIG_TIME 上。SIG_TIME 已包含 SIG；LONG_LINEAR 保持既有独立时序路线，本候选不叠加已完成的Sinc/MixStyle等机制。既有SIG_TIME3seed83.0393%为已暴露基准；本轮基础选择来自用户明确指令，不称新的独立盲选，参数/seed/父权重/停止规则不按target反馈调整。

最小4row：OLD_GATE_COS50-s01历史只读复用；SIG_TIME-s01同执行路径scratch控制；SIG_TIME_X3-s01三个独立6D SIG_TIME视图；SIG_TIME_MV_COV-s01独立6D SIG_TIME/MEAN_VAR/COV视图，t2四窗口，各descriptor升至28维，concat后单output projection到160、LN及boundedgain(.05初始/.2上限)加入真实reference.features。全为逐包运算，没有RX路由/类别参数/额外训练前向。后两pool各55968参数、fθ232033，单pool18976/fθ195041，默认W960保持normal joint；只删除W导出，保留cls_head的feature norms/gain与统计组合。

同路径的方向均值/方差可以从均值/完整协方差计算，不假称MEAN_VAR和COV独立信息；采用独立可学习局部投影作为不同表征视图，3SIG_TIME同参数控制区分更大容量与descriptor类型。控制并不保证固有descriptor秩一致。与原单SIG_TIME初始pool功能逐元素一致；原source/design3文件从已提交990ffabe0只读复用，无其他聊天工作树修改或旧SIG权重继承。

3新骨干各从零50/11100：33 label+17 old batch_neighbor pseudo gate、ownEMA.999、cos2e-4→1e-6、AdamW，真实L6300/U56700隐藏TX/V27000以及split392005原TX/RX/day/物理ID不变；标注训练池比例10%，V仅健康、无梯度/选模。NoExplicitCFO和内部补偿保持，无星地训练。新增模块进入原optimizer与EMA，全L/U/teacher/inference同feature policy。默认cos6x160scale30正常联合训练，不新建/另训/替换头；原生RX/day/channel ridge probe作用域禁用并核实实际调用一次返回disabled，不产生fitted reader。

新成本150model-epochs/33300updates，历史50单列复用，后续不从单seed最高target权重续训。预登记只首seed，未来02/03只依据固定源反证条件：三模型终轮LCE<=.5，实际新增参数更新和fθ导出/W独立，组合vsX3在固定L720差>=1e-6；资格先写源文件后汇总目标，无V微涨分/target候选重排。只有完整固定3seed均值才可84%达成，single row高分也不替代。

训练GPU CUDA上下文在native seed之前初始化，避免历史初次CUDA RNG混杂。真实initial_smoke状态核对三个native骨干/默认头一致、X3与组合trainable参数一致（COV的固定lift buffer不同，按定义保留）。固定final50后每新row先真实源L720 feature witness：独立仅fθ export和原joint160逐元素一致，W置零z不变，全部feature state观察不变、无头拟合；源cache V物化后丢弃，不前向、不读取U或target。然后自动clean168000预测，prediction固定后独立truthlast，每row18组CM/Macro-F1/RX/day/TX，4row672000/72metrics。

代码/config一个Git release HEAD；入口 python -u -m experiments.cvs_signature_stats50.dispatch；环境 ordinary N607 existing CVS-RFFI fullFP32/2threads; warm CUDA before native seed; private feature initialization RNG；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_signature_stats50_20261014_r01；不可变output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-sigtime-statistics-oldgate-cos-e50-clean-manysig-s01-m4-r01；共有capacity锁每GPU计所有owner最多3进程且空闲>=12000MiB，实际GPU/PID/配置/log-growth启动后记录。Stop：Any role/provenance/defaulthead/incompatible initialization/nonfinite/update budget/export/state/physicalID/prediction or truthlast technical failure stops new dispatch, preserves healthy existing rows/artifacts; no automatic rerun, target-derived early stop, parent choice or seed replacement。

本地descriptor数学/均值协方差关系/排列性质、真实adapter/默认头联合梯度及数值更新/teacher deepcopy/逐包一致性、真实额外模块导出W独立、协议负例、依赖/remotecompile PASS。一次新增直接P0/P1审查后提交Git并publish：单archive local/remoteSHA比对一次、remotecompile、真实旧同seed CP zeroIQ no-query smoke PASS后直接运行；没有签名/额外数据重验/报告hash或重复review门槛。

预期 preflight.json；launch.json；queue_state.json；pernewrow initial_smoke.pt/final_ssdg.pth/source_frozen.json/resolved_config.json/actual_native_args.json/initialization.json/50epoch/11100step/completion.json；pernewrow source_feature_witness.json；perrow default prediction/default_frozen/provenance/complete.json；perrow scores/metrics.json and completion.json；source_features.json；source_qualification.json；analysis.json；completion.json；failure.json only if failed。数据/权重/大型prediction/完整原日志留既定产物目录，小型报告/聚合结果和实际实现Git上传并独立核对远端OID。目前实现和局部验证完成，尚未发布或有新性能结论。

唯一独立审查发现2项直接P1：private CPU fork调用torch.manual_seed会重置已初始化CUDA RNG；未被当前native消费的labeled_context导致主L计数0。均已定点修复：CPU default_generator独立seed；实际root.forward计数，仅训练且合法y_tx主L进入，U None/eval不计，仍执行原forward，deepcopy绑定自身。真实CPU/CUDA RNG与native主L/U/eval/teacher合成定点检查PASS。正在完成该两处一次定点确认；旧框架不重审。

唯一完整P0/P1审查与原两项P1的一次定点确认均 PASS。环境重置后仅恢复未完成的该两项定点确认；未重复全面审查。实际CPU/CUDA RNG与主L/U/eval/teacher绑定合成检查通过，其余新执行路径无直接缺陷。接下来提交代码/config Git HEAD并发布，启动后记录实际GPU/ownedPID/CWD/argv/配置/log-growth，不再增加门槛。

实际发布 VERIFIED，执行代码 eda57144de34d479288f42d14c037eb7802c1ecd，dispatcher PID2051754，三个新模型 GPU4/5/3（SIG_TIME/X3/MV_COV），epoch2/2/2，配置与owned进程/CWD/argv/log-growth独立核实。原生另训probe禁用=true；newpool18976/55968/55968与设计一致。旧baseline行已复用评分，三个新模型固定50，不热改。单archiveSHA与remotecompile、真实同契约CP no-query smoke PASS。新性能结果尚未产生；详见[启动证据](launch_evidence.json)。


首种子四行均已评分和独立复算，组合83.1220%未优于单分支83.5756%/X3 83.8333%；完整三种子尚在运行。详见 [完整首种子分析](final_analysis.md)。
