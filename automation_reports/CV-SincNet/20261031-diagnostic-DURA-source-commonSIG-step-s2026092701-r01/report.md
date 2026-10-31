# 波形与时序签名共同下降：固定源IQ步进对照

- run_id：`20261031-diagnostic-DURA-source-commonSIG-step-s2026092701-r01`
- group_id：`phase1-dura-source-common-IQ-signature-step`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定defaultDURA/source512×01/20，同球四步共同IQ-SIG方向，对比原SIG损失缩步；源配对clean，只读原三组控制输出，0G/TX-V-target。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

源诊断完整结果和固定科学判定FAIL见analysis.md/results.json，无新G或正式测试。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 固定共同步进与原控制

检查IQ梯度范数cap并不保证签名方向同向这一具体问题。固定defaultDURA/source512（128 L_s/384 U_s）×缓存01/20，复用原signatureLossStep的native、同球最近clean IQ、SIG-only三组源IQ/特征/签名/原头输出。新算法仅四步：单位gIQ＋单位gSIG为方向，取非负精确IQ线最小步长与SIG loss/方向导数的较小者；每步最多0.25%原IQ RMS，总投影到1%原IQ RMS球。零梯度、不可定义或相反方向不强迫步进，不搜索步数/步长/权重。

沿方向的精确IQ最小步长和凸投影保证最终IQ误差不超过原anchor（浮点容差单列）；SIG只有当前一阶下降关系，不声称非线性实际误差必然单调，更不是G参数梯度、全训练轨迹或真实身份保证。完整逐步IQ、SIG导数和loss保存到私有数组，供独立计算而不再调用F。

两个缓存均须相对旧SIG-only至少多2条clean F选择一致，每源RX和clean不损失、每非clean景最多损失1条，平均SIG相对误差≤0.9native、每条IQ误差≤native（3e−5相对＋1e−10绝对容差）。通过仅支持另登记从零单阶段E30研究，不自动训练；失败不调整阈值。源配对clean不可用于部署，一致代理不是真实TX，旧PA/步进/RXspan FAIL均保持。

3项聚焦检查与完整47基底实际模块入口PASS，唯一P0/P1 PASS。本地直接入口因缺少旧基底包而失败，在已验证的47基底路径下入口通过，候选代码未为此添加兼容分支。首步真实2Ls F分量/加权SIG/原头和实际共同导数/IQ有限步PASS后立即继续。CPU2/空CUDA；主完整F1024/前缀4096/VJP4096/原头4096，smoke另完整F2/前缀2/VJP2/原头2；0G读取/前向/模型优化/TX/V/target。继承原F50/11100知识与旧缓存成本，F冻结；一次preflight/归档SHA/远端编译/PID-CWD-argv-CUDA日志核实及全数组复算，默认保持。

多目标优化研究提供问题背景，本次具体归一化角平分方向＋IQ精确线最小源步进不是其算法复现，也不继承论文的性能结论：[Sener与Koltun，2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)，[Yu等，2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。


## 完整源结果

VERIFIED / ANALYZED_SOURCE_ONLY，科学判定FAIL。固定defaultDURA/source512（128L/384U）×01/20，四步共同IQ-SIG方向与旧SIG-only对照。一致数501/504与旧SIG-only501/504相同，IQ误差相对旧控制下降6.417819%/6.112685%，但源RX保护失败、第一缓存场景保护也失败。IQ输出空间负余弦仅0.927734%/0.634766%，cap0.5合成方向对SIG的一阶预测增加比例均为0，未支持此处大规模方向冲突这一假设。所有逐包IQ相对anchor保护通过，非线性SIG误差没有自动同等改善；不启动该G、不改阈值。CPU2主循环5.335702秒，独立复算0.491907秒，完整4096step-record/1024观察/4096原头特征行/52组通过；独立最大IQ步进差1.19456782244e-07。主完整F1024/前缀4096/VJP4096/原头4096，真实2Ls另F2/前缀2/VJP2/原头2；0G加载/前向/模型更新/TX/V/target。源clean F选择代理不是TX准确率，输出空间结论不是G参数或全轨迹结论。默认DURA/完整目标ACTIVE保持。

详见[分析](analysis.md)，PID1757624自然结束，旧控制与负结果保留。
