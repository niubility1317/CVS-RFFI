# DURA 与单阶段对照的源输出整体相位误差分解

- run_id：`20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01`
- group_id：`phase1-dura-source-output-phase-decomposition`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定source512×clean/缓存01/20×2冻结G；仅相位oracle不拟合幅度/形状；同F特征/原头保持、CPU2/0更新/TX/V/target。

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


## 最小研究说明

仅source固定512（L128/U384），比较原noVar cf40和默认R69，两模型都不更新。每模型先真实2Ls同F checkpoint smoke，再完成clean和原R07缓存01/20三个视图。oracle使用原clean的复相关，只给输出乘单位复数；绝不拟合幅度或改变形状。零相关相位不唯一，记录valid=False，identity仅中性代表，不冒充已估得相位。

相位可消除误差是原IQMSE的精确分项，不是新的参考：设c=mean(r·conj(x))，则原误差与相位对齐误差之差为2(|c|−Re(c))/Ex。该oracle依赖clean，不可部署，也不是TX准确率上界。F指标是同F自身clean选择的一致，不读逐样本TX/V/target。原F知识和G祖先身份保留，当前神经更新0。

仅noVar可支持下一研究：两个缓存视图分别相位有效≥480/512、可消除误差占总IQ误差≥10%、教师选择净增≥6/512、七个源场景均不下降；另clean控制不下降。任一失败不立项该phaseweight训练，不放宽门槛。通过仅允许另登记单阶段scratchE20，将IQMSE中相位分项权重从1变2，原clean和其余损失保持；不部署oracle，不忽略相位，不复活统一相位旋转或公共µ。默认DURA和完整逐景3pp/clean/效率目标不变。


本地9项相关检查通过，唯一独立P0/P1审查PASS；N607预检核实身份/路径。使用既有CPU2环境与原VALIDATED_ONCE源句柄，源码/config以本次Git commit固定，真实F smoke在launcher首步直接继续。
