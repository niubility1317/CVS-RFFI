# PAMF第三阶段：统一星地G10开发观测

- run_id：`20261029-phase3-star-g10-pool-rx24-1-r01`
- group_id：`pamf-phase3-star-only-g10-input`；类别：`data`；阶段：`Phase3-star-only-G10-development-data`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED／VERIFIED_DATA_ONLY

## 目的与对照

Preserve exact RX24-1 U384/Q864 physical roles; create one fixed MILKR G10 MIX01 overlay per record, no clean model input, model or scoring.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

G10数据已完整构建，零模型/预测/评分。实际数据与证据见下方结果。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 发布前冻结

实际code/config commit `3f24f8016f5586f096bbaa8d83f221b397dbbeb8`；原Phase3完整目标保持。12tests/47subtests PASS，参考MILKR的MIX/三pure多seed逐值一致，真实合成channel smoke通过，唯一独立P0/P1 PASS。无F/G加载，checkpoint smoke和N607发布不适用于本机CPU数据作业。

命令：`"C:/Users/lh594/.conda/envs/ssr-gpu/python.exe" -X utf8 -B -u -m experiments.pamf_phase3_open_world.star_development_pool --config configs/pamf_phase3_star_g10_pool_20261029.json --output "E:\type10-7\local_artifacts\pamf_phase3_star_g10_pool_20261029_r01\pool"`；CWD：`E:\type10-7\github_publish\CVS-RFFI-repo`；Python：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`；新输出：`E:\type10-7\local_artifacts\pamf_phase3_star_g10_pool_20261029_r01\pool`。缺匹配Full/旧role、非法形状或重复/丢失canonical、非有限观测或输出碰撞即保留partial，不自动重启/换样/覆盖；无target performance阈值选择。预期artifact和source/target权限见experiment.json；实际PID/CWD/argv/日志/完整产物随后独立核实。

## 实际结果

**VERIFIED／VALIDATED_ONCE／ANALYZED_DATA_ONLY**。实际代码/config `3f24f8016f5586f096bbaa8d83f221b397dbbeb8`，U384/Q864，1248个原物理记录各增加一份本MIX01固定星地实现；仅两已索引Full成员，未新增独立物理样本。旧U/Q角色保持，旧received/truth/scene/score读取均0；新F/G加载、训练、预测、评分、正式RX24-18读取和GPU新增均0。

| 角色 | 实际场景 | 记录数 | 实际仰角°范围 |
| --- | --- | ---: | --- |
| discovery | practical_high | 17 | 45.7103 至 76.9954 |
| discovery | practical_mid | 27 | 30.7488 至 43.6544 |
| discovery | practical_low_suburban | 158 | 10.0764 至 29.6692 |
| discovery | practical_high_urban | 20 | 45.4529 至 76.2168 |
| discovery | practical_mid_urban | 32 | 31.3383 至 44.7697 |
| discovery | practical_low_urban | 130 | 10.0724 至 29.9985 |
| evaluation | practical_high | 39 | 45.1953 至 78.1206 |
| evaluation | practical_mid | 67 | 30.0146 至 44.7757 |
| evaluation | practical_low_suburban | 320 | 10.0422 至 29.6956 |
| evaluation | practical_high_urban | 54 | 45.2506 至 77.8500 |
| evaluation | practical_mid_urban | 60 | 30.5614 至 44.9584 |
| evaluation | practical_low_urban | 324 | 10.0629 至 29.8799 |

六景按参考G10概率逐记录抽样，有限样本计数无需等于配额，不按真实类别或角色重平衡。公开每角色精确三文件、phase3_min_v1；原始canonical、场景/仰角/seed和truth留私有builder/scorer侧。

discovery capsule=`p3dev-star-g10-6e1292b62ab09286ebf1a0df`，split=`83c5cc1fe2f2a1512dbc1287`。

evaluation capsule=`p3dev-star-g10-71a5a0dec23ac6ccc3b479c2`，split=`05de65cd4818f3e5e33ced5c`。

完整读取本次11行stdout，无异常；已独立捕获真实子进程PID/命令与CLI的CWD，日志增长与终态artifact由build_observations/build_post_state记录；不把启动返回值当完成。此次只证明输入数据就绪，新的G10拒识/适应/注册性能尚未测定，Phase3目标保持ACTIVE。原clean与三等量压力景成绩不迁移到本设置。

证据：`local_artifacts/pamf_phase3_star_g10_pool_20261029_r01/{pool/complete.json,pool/builder_report.json,analysis_summary.json,build.log,build_observations.jsonl,build_post_state.json}`；无IQ/私有truth/原始日志入Git。
