# PAMF第三阶段：星地G10相对距离与无标签注册

- run_id：`20261029-phase3-star-relative-s2026092701-r01`
- group_id：`pamf-phase3-star-only-relative-geometry`；类别：`diagnostic`；阶段：`Phase3-star-only-G10-open-world-development`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Original Phase1 frozen singleF0; sourceL/V star-only G10, MD/RMD stats andknown-risk calibration; P0/P1 fourstate futureQ independentU-only scoring.

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

## 实际发布前固定

实际代码/config `a391d7ae8baeb5b772ebb93854c92606cd0aa95a`。source L/V原始记录先G10、单F0/default6头与固定G冻结；七分数主RMD，不用旧E51/clean状态；sourceL拟合全部几何/metric/尺度，V只42组globalτ。U384/Q864新G10，P0/P1四态全部固定后独立truth-last，U-only匿名映射，不读取真K。

29项unique tests/33subtests PASS，source9项含旧F0实际schema兼容回归（修复后只定向3项，无全套重复）；唯一独立P0/P1 PASS。一次N607 preflight已核rawManySig/原L/Vmetadata/原F0源contract与checkpoint存在、matchingG10handles及实际显存。源/目标真实checkpoint合成星地smoke仍为各job首步，不是额外授权。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_relative_launch --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-relative-s2026092701-r01-release/runtime.json`。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-relative-s2026092701-r01-release`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。单owner、新run/outputroot见experiment.json，GPU0–7动态选free≥4096MB、串行一GPUworker，每卡最多新增2，不修改其他owner。顺序source→P0pred/score→P1pred/score；dataset/weight/log/rawprediction不提交。技术角色/继承/shape/nonfinite/outputcollision/incomplete预测故障保留partial，不盲重启；科学失败、零簇或core不足不改配方/重跑。所有source状态在U前、U状态在Q前冻结；正式RX24-18不访问，不扩seed。预期complete/聚合状态/四态pred及margin/Udiag/独立score/完整日志与PID/CWD/argv/GPU/log-growth由独立post-state核实。
