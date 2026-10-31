# 中层签名辅助监督：从零单阶段30轮

- run_id：`20261031-phase1-DURA-midSIG-E30-s2026092701-r01`
- group_id：`phase1-dura-middle-signature-aux-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留物理四项与原onlineSIG指导，仅增加G粗恢复时域encoder第2层的源cleanSIG回归，权重.01；投影仅训练执行。完整G287591/新增20640，复用原40.32MB源特征，完整七景。

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


## 固定中层签名辅助任务

原SIG指导保留，仅增加G粗恢复时域编码器第2层（B×16×128）的源clean签名回归：8格平均池化后Linear128→160，权重0.01。它是连续特征投影，不是TX/RX分类头，不更换原F/defaulthead。原IQ1/priorIQ0.1/STFT0.05/actionprobe0.05及onlineSIG的IQ梯度cap0.5保持。aux普通梯度进入粗恢复的频谱路径与时域浅中层，不包含在onlineSIG输出空间cap中；全G仍clip1。

G全部参数从零、单阶段30轮/7410步，原主体初值逐项匹配SIG对照47591。投影CPU-only fork_rng、小随机权重1e−3/bias0，不扰动原主体或CUDA探针。新增20640参数、完整G287591，包含投影的参数与存储全部计入。投影仅在训练临时hook中调用，原G和优化推理路径不执行它；不是通过删optimizer宣称压缩。

同原source63000与20份received缓存后重复01至10，L189000/U1701000、model/CUDAprobe2026092701/npshuffle2026103004/AdamW.001warm247cos2e−5/WD1e−4保持。复用原source clean SIG数值40.32MB，无新缓存F前向或复制；历史63000前缀/0.847077秒和F50/11100监督知识保留，属于F-guided，不称F-free。旧G权重不读取，缓存不依赖旧G训练或目标结果。原F50/默认头冻结，无直接TX/V/target、classloss或另训分类头。

6项聚焦检查、实际factory参数/初值/别名核验和实际入口PASS，唯一P0/P1 PASS。首步真实2Ls检查实际aux对频谱、时域浅中层及投影的梯度；复用cache对真实F、原onlinecap、全部G模块、strictE0重载、无推理aux调用、原生/优化输出及原F原头state。smoke G8/前缀F4/完整F2/F输入VJP2，无新cache和TX/V/target。PASS立即继续30轮及同卡profile、source1800×7、formal168000×7，每role全7预测固定后独立CPUtruth-last，源差也不取消正式测试。

本版是源特征机制与G梯度路径的独立检验，不按目标TX/RX分数设层位/权重，不继承旧G或修改默认DURA。每六星地≥137216正确/81.6761904762%、clean≥142256；R17逐景最多损失1680与旧低城区70%另列。完整质量、参数/存储/延迟/显存/缓存和训练成本全部报告，目标ACTIVE。
