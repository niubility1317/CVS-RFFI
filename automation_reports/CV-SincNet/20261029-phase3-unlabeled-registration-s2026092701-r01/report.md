# PAMF第三阶段：冻结身份的无标签适应与匿名注册四态闭环

- run_id：`20261029-phase3-unlabeled-registration-s2026092701-r01`
- group_id：`pamf-phase3-frozen-unlabeled-adaptation-registration`；类别：`diagnostic`；阶段：`Phase3-unlabeled-adaptation-anonymous-registration-development`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen originalF0 identity/defaulthead plus E51 augmented rejection. Source L density/anchors and knownV calibration, independent condition U bias/discovery/anonymous registry, then frozen four-state futureQ and truth-last U-only permutation scoring.

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

## 实际发布前冻结

# Phase3 四态无标签适应／匿名注册最小预登记

run：20261029-phase3-unlabeled-registration-s2026092701-r01；commit：a6679caca686c9941a5b948764c1dd0a9330e882。
主状态 DA1_REG1::mlp16_oe；固定四态 DA0_REG0, DA1_REG0, DA0_REG1, DA1_REG1。
科学方法／矩阵见 configs/pamf_phase3_unlabeled_registration_20261029.json 与 docs/PAMF_PHASE3_UNLABELED_REGISTRATION_PLAN_20261029.md，不改旧goal。

- 环境：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python；CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-unlabeled-registration-s2026092701-r01-release
- 单一launch owner：codex-pamf-phase3-unlabeled-registration-20261029；新不可覆盖run：/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-unlabeled-registration-s2026092701-r01
- 命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.unlabeled_registration_launch --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-unlabeled-registration-s2026092701-r01-release/runtime.json
- GPU：0–7动态选择，实际free≥4096MB，每卡最多新增2，本run最多1GPU worker；不改其他owner。
- 顺序：source → C0 Ufit/四态Qpred → C0独立U/Qscore → P1 Ufit/四态Qpred → P1独立U/Qscore。
- source：原F0＋E51／aug读出＋G冻结；sourceL每condition6300一view，V27000四view compact；无新NN拟合。
- 目标：每condition U384/Q864物理互斥，同1248原记录跨condition复用；未知类别数／配额／TX不进predict。
- U truth仅打包到release/score_truth/<condition>/U.npz；Q truth复用旧只读scorer路径。预测配置绝无truth。
- 门槛：clean95/star90且FRR5／原F0open2pp；新类future宏80／注册旧类损失2pp，完整逐scene/TX四态报告。
- 停止：身份/角色/非法truth/Q拟合/非有限/碰撞/无法合法预测；保留partial，不重试救分。0cluster/core不足不视为技术失败。
- 实际命令、路径与artifacts见runtime.json、staged_commands.json、filled spec；当前脚本未登记/复制truth/发布/启动。

聚焦验证：core10tests/15sub、source8/8sub、target4/4sub、score7/6sub、launch2/5sub，合计31tests/38sub PASS。实际断言与协议负测已完成；唯一独立P0/P1审查PASS（本run review.md），不重复全审。N607单次preflight已PASS，预检只读、未停止其他owner；正式模型smoke仍为launcher源阶段第一步，PASS后自动继续。
