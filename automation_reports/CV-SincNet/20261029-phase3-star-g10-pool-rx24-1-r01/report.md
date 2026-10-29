# PAMF第三阶段：统一星地G10开发观测

- run_id：`20261029-phase3-star-g10-pool-rx24-1-r01`
- group_id：`pamf-phase3-star-only-g10-input`；类别：`data`；阶段：`Phase3-star-only-G10-development-data`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Preserve exact RX24-1 U384/Q864 physical roles; create one fixed MILKR G10 MIX01 overlay per record, no clean model input, model or scoring.

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

## 发布前冻结

实际code/config commit `3f24f8016f5586f096bbaa8d83f221b397dbbeb8`；原Phase3完整目标保持。12tests/47subtests PASS，参考MILKR的MIX/三pure多seed逐值一致，真实合成channel smoke通过，唯一独立P0/P1 PASS。无F/G加载，checkpoint smoke和N607发布不适用于本机CPU数据作业。

命令：`"C:/Users/lh594/.conda/envs/ssr-gpu/python.exe" -X utf8 -B -u -m experiments.pamf_phase3_open_world.star_development_pool --config configs/pamf_phase3_star_g10_pool_20261029.json --output "E:\type10-7\local_artifacts\pamf_phase3_star_g10_pool_20261029_r01\pool"`；CWD：`E:\type10-7\github_publish\CVS-RFFI-repo`；Python：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`；新输出：`E:\type10-7\local_artifacts\pamf_phase3_star_g10_pool_20261029_r01\pool`。缺匹配Full/旧role、非法形状或重复/丢失canonical、非有限观测或输出碰撞即保留partial，不自动重启/换样/覆盖；无target performance阈值选择。预期artifact和source/target权限见experiment.json；实际PID/CWD/argv/日志/完整产物随后独立核实。
