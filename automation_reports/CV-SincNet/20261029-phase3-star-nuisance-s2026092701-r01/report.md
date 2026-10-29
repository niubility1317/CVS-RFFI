# PAMF第三阶段：星地域残差投影源端证伪

- run_id：`20261029-phase3-star-nuisance-s2026092701-r01`
- group_id：`pamf-phase3-star-nuisance-source-falsification`；类别：`diagnostic`；阶段：`Phase3-source-star-open-set`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；实际完成结果见下。

## 目的与对照

Source-only fixed control/RX/RXday MD/RMD; L-only protected nuisance projection, known-only V risk calibration, G10 heldout-device score.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

实际作业已完成，科学结果与证据边界见文末追加。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 实际发布前固定

代码/config commit fccbde38a407891db5acce776d6ad7af891f1341；固定源端三变体、MD/RMD、P0/P1共12列，唯一主候选rxday::RMD::P1。全部神经0更新，原六类默认头和G冻结；source L拟合，known V仅42组阈值，novel public在冻结后读取，全部预测完成后独立truth评分。新联合指标定义见 docs/PAMF_PHASE3_METRICS_20261029.md。

聚焦组件共34项unique测试：metrics6、data6、core7、runner/score13、launch2，子项另记；source owner最后仅状态名称改动后定向1项通过。data与模型候选各一次独立P0/P1 PASS。数据3150已VALIDATED_ONCE，原ID/物理角色保持；首次真实checkpoint合成星地smoke由source job首步执行，PASS即继续，不新增许可。

一次N607 preflight已通过：原source合同/原F0/G与metadata/raw路径存在，目标句柄0，实际各卡余量已记录；动态选择free至少4096MB，串行一个本任务GPU worker，不停止其他owner。archive单次SHA比对和remote compile在发布时执行；不按source分数调参或改候选，不追改历史G10/clean实验。科学失败完成评分并NO_PROMOTION，无target访问；通过只表示下一冻结目标小矩阵的source eligibility。

## 2026-10-29 源端 RX/day 硬投影证伪结果

`20261029-phase3-star-nuisance-s2026092701-r01` 两jobs完成；主 `rxday::RMD::P1`：**NO_PROMOTION**，总目标 **ACTIVE**。

| 候选 | A_K% | R_U% | H_OS% | macroOSCR% | C_K% | F0损pp | 源gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control::MD::P0 | 75.1370 | 2.7937 | 5.3870 | 58.6502 | 99.0519 | 0.6741 | False |
| control::MD::P1 | 90.7037 | 0.8889 | 1.7605 | 77.2587 | 99.3296 | 0.2778 | False |
| control::RMD::P0 | 75.3778 | 4.0000 | 7.5969 | 62.5673 | 99.0852 | 0.4333 | False |
| control::RMD::P1 | 90.4519 | 6.4444 | 12.0317 | 78.1555 | 98.8370 | 0.5296 | False |
| rx::MD::P0 | 75.2815 | 4.6667 | 8.7885 | 61.4871 | 99.0815 | 0.5296 | False |
| rx::MD::P1 | 90.5963 | 1.9048 | 3.7311 | 77.6677 | 99.1037 | 0.3852 | False |
| rx::RMD::P0 | 75.2481 | 5.1429 | 9.6277 | 62.5699 | 98.7296 | 0.5630 | False |
| rx::RMD::P1 | 90.6667 | 2.0952 | 4.0958 | 77.1439 | 99.2111 | 0.3148 | False |
| rxday::MD::P0 | 75.4593 | 3.9365 | 7.4827 | 62.8788 | 99.2444 | 0.3519 | False |
| rxday::MD::P1 | 90.6333 | 2.1270 | 4.1564 | 77.9211 | 99.1963 | 0.3481 | False |
| rxday::RMD::P0 | 75.3593 | 3.8413 | 7.3099 | 62.4770 | 98.9741 | 0.4519 | False |
| rxday::RMD::P1 | 90.6296 | 2.3492 | 4.5797 | 77.2395 | 99.1741 | 0.3519 | False |

只L6300拟合投影/Gaussian/score scale，V27000额外known标签只作42组τ校准，novel3150固定预测后独立评分；target输入/U_s IQ/OE训练IQ/E51/NN更新均0。主candidate known42保护=True、unknown必需12组90%=False。
33300known为同physicalID/同episode的确定性重建；novel为旧3150物理记录的新G10观测，已有源开发暴露、同sourceRX，不能当新物理或DG确认。单F0＋G499944 NN参数，72900F/36450G计算，聚合状态另计。
主对control的ΔA_K/ΔR_U/ΔH_OS/ΔOSCR/ΔAUROC=0.1778/-4.0952/-7.4520/-0.9160/-1.0847pp；不按单项换主。资源samples=[(2849941, 798), (2849941, 832)]非peak，终态owned/live0、foreignstop0。
主分数AUROC82.7789%低于同时95%接收与90%拒识所需85.5%必要界；known保护已过，问题是源开放集分离，而非闭集head漂移。主P1五branch硬删除rank12/15/9/5/3虽保留preunit类间span，reunit后fused类均值仍变化31.29%；该机制未通过源端证伪，不推进target。
完整逐RX/scene/TX、42组保护、实际投影rank/能量与类间损伤、计数及权限边界见 `docs/PAMF_PHASE3_STAR_NUISANCE_RESULTS_20261029.md`。无target验证/DA/注册结果，不宣布Phase3完成；后续假设未实施。
