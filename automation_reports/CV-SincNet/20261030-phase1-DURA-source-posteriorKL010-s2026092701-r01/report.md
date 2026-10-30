# DURA 源 F 连续后验保持单阶段对照

- run_id：`20261030-phase1-DURA-source-posteriorKL010-s2026092701-r01`
- group_id：`phase1-dura-source-F-posterior-preservation`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同冻结原F50/defaulthead的源特征映射成6D连续后验，KL(clean||restored)权重.1；G从零E20/4940，无硬伪TX/真实TX标签/F更新，自动全七景。

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


## 固定方法与权限

在已授权源F特征约束范围内，将同一冻结F的160维源特征经原默认头映射为连续6维后验，使用KL(p_clean||p_restored)，温度1、权重0.1。没有硬argmax标签或真实TX标签，不更新F/原头，也不新增判别器。G仍从零单阶段E20/4940；原noVar基底和普通源配对不变，不混入频谱增强或信道重配。

该方案继承F既有源监督知识，必须标为F-guided，F上游50轮成本另列。0.1与余弦对照同数值，损失单位和梯度强度并不相同，不作等强度声明。选择该固定方案的源依据是既有特征误差与教师选择保持的差异，不按目标场景/类别调权。实际是否改善以完整七景独立评分为准，默认DURA保持。
