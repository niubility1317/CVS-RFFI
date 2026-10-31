# 固定已训练恢复器的特征条件路径：源域匹配干预

- run_id：`20261031-diagnostic-DURA-source-priorContext-oracle-s2026092701-r01`
- group_id：`phase1-dura-source-fixedweight-prior-context-oracle`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定SIGcontextPrior final30权重，源512×缓存01/20比较正常、关闭调制、cleanSIG条件及全512循环移位1的cleanSIG条件；主cleanSIG utility，0训练/TX-V-target。

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


## 固定权重的源域因果路径检验

上一轮主版与匹配辅助版分别训练，差值包含训练轨迹和解码条件路径的共同作用。本诊断固定 main SIGcontextPrior 的 final30/7410、344295 参数和原 F50，严格读取同源数据契约与配置，比较相同源 IQ 下四种读出：正常预测条件、关闭两个条件层输出、在条件层输入提供本条 clean SIG，以及提供全512循环移位1条的 clean SIG。所有预测头仍实际执行；不修改 G 输入、模型权重或分类头，不训练，不访问 TX/V/target。

固定 source512（128L/384U）、缓存01和20，共1024次观察、512个物理记录。clean SIG 参照来自已验证的 F-only 63000×160/40.32MB 缓存；完整 F 特征和 clean F 选择参照只读既有 phase oracle 缓存第0行。SIG 与完整 F 特征虽然均为160维，但含义不同，不互换。缓存知识和 F50/source4df7/50轮11100步、G30轮成本披露，不称 F-free。

主检验为 clean SIG 条件的源用途：两个视图均需比正常预测多至少4/512个 clean F 选择一致、比循环移位对照多至少2个；5个RX净差非负、clean场景非负、每个非clean场景净差至少−1、全 F 特征余弦误差不变差。关闭调节仅为描述，不能替换主检验。FAIL 保留，不改阈值、样本、对照或自动加训；PASS也只支持另行学习假设。

源 clean F 选择可能不是真实 TX；这批源数据曾用于 G 训练。将真实 clean SIG 放入已适配预测特征的条件层，可能偏离训练输入分布。因此该理想参照不可部署，不是目标收益、物理恢复极限或任何更强预测器的普适有效性证明。循环移位保持源特征池的边际分布，不保持逐条范数或 IQ 修正能量，也不假设供体一定来自其他 TX。

3项聚焦检查与实际344295模型/CLI/CPU输出边界/四路径/条件值/非零关闭效应/异常hook恢复/state和grad检查通过。唯一独立P0/P1审查在发布前完成。首步真实2条非clean L_s，严格 final30 加载、源 SIG 与 clean F 参照、实际条件输入、关闭输出、恢复原IQ和原分类头 smoke：G12/F4、0更新，PASS直接继续。主循环G4096/F4096、CPU2，无GPU/真值评分进程。全部IQ/F/预测特征/实际条件输入输出/W和bias保留私有NPZ，Git只保存聚合证据；独立NumPy复算144个分组case、实际线性条件层、零输出、输入置换及规则，不新增NN前向。默认DURA和完整六景3pp/clean/同质量效率目标保持。


## 完整源诊断结果

VERIFIED / ANALYZED_SOURCE_ONLY，主条件PASS。相同权重正常487/500一致，关闭481/497，本条clean SIG498/505，循环移位480/498；144分组case和实际条件层独立复算通过。详见[完整分析](analysis.md)与results.json。CPU2主循环14.107593秒，G4096/F4096、另smoke G12/F4，0TX/V/target/更新；进程自然退出。该源理想参照不可部署或认证恢复极限，不自动训练。
