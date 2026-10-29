# PAMF第三阶段：星地G10相对距离与无标签注册

- run_id：`20261029-phase3-star-relative-s2026092701-r01`
- group_id：`pamf-phase3-star-only-relative-geometry`；类别：`diagnostic`；阶段：`Phase3-star-only-G10-open-world-development`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（ANALYZED_DEVELOPMENT_ONLY）；五stage全部完成，科学结果见下。

## 目的与对照

Original Phase1 frozen singleF0; sourceL/V star-only G10, MD/RMD stats andknown-risk calibration; P0/P1 fourstate futureQ independentU-only scoring.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

五stage已经完成，P0/P1四态独立评分结果及边界见本文末尾实测追加。
本轮没有新神经训练；源统计、预测与独立评分均完成，两个path均NO_PROMOTION。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 实际发布前固定

实际代码/config `a391d7ae8baeb5b772ebb93854c92606cd0aa95a`。source L/V原始记录先G10、单F0/default6头与固定G冻结；七分数主RMD，不用旧E51/clean状态；sourceL拟合全部几何/metric/尺度，V只42组globalτ。U384/Q864新G10，P0/P1四态全部固定后独立truth-last，U-only匿名映射，不读取真K。

29项unique tests/33subtests PASS，source9项含旧F0实际schema兼容回归（修复后只定向3项，无全套重复）；唯一独立P0/P1 PASS。一次N607 preflight已核rawManySig/原L/Vmetadata/原F0源contract与checkpoint存在、matchingG10handles及实际显存。源/目标真实checkpoint合成星地smoke仍为各job首步，不是额外授权。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_relative_launch --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-relative-s2026092701-r01-release/runtime.json`。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-relative-s2026092701-r01-release`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。单owner、新run/outputroot见experiment.json，GPU0–7动态选free≥4096MB、串行一GPUworker，每卡最多新增2，不修改其他owner。顺序source→P0pred/score→P1pred/score；dataset/weight/log/rawprediction不提交。技术角色/继承/shape/nonfinite/outputcollision/incomplete预测故障保留partial，不盲重启；科学失败、零簇或core不足不改配方/重跑。所有source状态在U前、U状态在Q前冻结；正式RX24-18不访问，不扩seed。预期complete/聚合状态/四态pred及margin/Udiag/独立score/完整日志与PID/CWD/argv/GPU/log-growth由独立post-state核实。

## 2026-10-29 G10 星地单 F0 相对距离四态结果

`20261029-phase3-star-relative-s2026092701-r01` 完成5jobs与独立U/Q truth-last评分；两 path 均 NO_PROMOTION，总目标 **ACTIVE**。新source只消费原L6300/V27000的G10观测，唯一F0＋G全冻结、神经0更新，无E51/OE/clean模型输入；σ只来自L，V只known-only42组τ校准。

| path | 状态 | 旧接收% | 旧 open% | F0 损 pp | REG0 拒识% | 未来新 macro% | H% | 注册 FRR% | later 拒识% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | DA0_REG0 | 99.1319 | 56.7708 | 0.3472 | 8.6806 | N/A | N/A | N/A | 13.5417 |
| P0 | DA1_REG0 | 99.1319 | 56.7708 | 0.3472 | 7.6389 | N/A | N/A | N/A | 14.5833 |
| P0 | DA0_REG1 | 99.1319 | 56.7708 | 0.3472 | N/A | 0.0000 | 0.0000 | 2.2135 | 13.5417 |
| P0 | DA1_REG1 | 99.1319 | 56.7708 | 0.3472 | N/A | 0.0000 | 0.0000 | 1.6927 | 14.5833 |
| P1 | DA0_REG0 | 97.2222 | 71.0069 | 1.9097 | 3.1250 | N/A | N/A | N/A | 4.1667 |
| P1 | DA1_REG0 | 97.5694 | 71.1806 | 1.7361 | 2.7778 | N/A | N/A | N/A | 4.1667 |
| P1 | DA0_REG1 | 97.2222 | 71.0069 | 1.9097 | N/A | 0.0000 | 0.0000 | 2.7344 | 4.1667 |
| P1 | DA1_REG1 | 97.5694 | 71.1806 | 1.7361 | N/A | 0.0000 | 0.0000 | 2.3438 | 4.1667 |

P0：REG0 RMD拒识7.6389%、旧接收99.1319%、open56.7708%；注册后新macro0.0000%、注册FRR1.6927%、later拒识14.5833%、F0总损0.3472pp。adaptation=False、registration=False；scene缺类用N/A/coverage，不填0或认证。
P1：REG0 RMD拒识2.7778%、旧接收97.5694%、open71.1806%；注册后新macro0.0000%、注册FRR2.3438%、later拒识4.1667%、F0总损1.7361pp。adaptation=False、registration=False；scene缺类用N/A/coverage，不填0或认证。

两path同一RX24-1 U384/Q864、同1248物理G10记录，跨日但非source未见日期；正式RX24-18未读。F0+G499944 NN参数，额外metric统计另计；source33300物理、66600F/33300G计算曝光。独立GPU采样[(2818487, 404), (2818487, 798)]是采样非峰值；终态owned/live为空与foreignstop0以observed_state核实，pipeline119.498955s非逐包时延。全部方法/scene/TX/簇组成与计数边界见 [完整结果](../../../docs/PAMF_PHASE3_STAR_RELATIVE_RESULTS_20261029.md)。

两path全局旧类接受/同输入F0损失单项通过，但未知拒识和未来新类80%均失败；P1 high urban/low urban 的F0损伤4.1667/3.5755pp超限，high suburban缺19-15而不能完整认证。四态注册K全0，P0/P1新类184/188条错接为旧类，故低注册FRR不代表成功。DA1 RMD AUROC .774185/.652507低于.855必要界；P1 RMD较MD排序低7.7498pp，不能仅平移τ解决。

下一假设（未实施）：只在源侧区分RX/day扰动和身份方向，检验能否改善拒识排序及候选密度支持；当前没有可评估的已注册原型，不归因于已证实的跨日半径失败。保留全部科学目标，不用Q调参或提前完成总Phase3。
