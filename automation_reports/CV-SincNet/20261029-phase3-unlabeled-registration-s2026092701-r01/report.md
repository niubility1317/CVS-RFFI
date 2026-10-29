# PAMF第三阶段：冻结身份的无标签适应与匿名注册四态闭环

- run_id：`20261029-phase3-unlabeled-registration-s2026092701-r01`
- group_id：`pamf-phase3-frozen-unlabeled-adaptation-registration`；类别：`diagnostic`；阶段：`Phase3-unlabeled-adaptation-anonymous-registration-development`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（ANALYZED_DEVELOPMENT_ONLY）；五个 stage 全部完成，C0/P1 均 NO_PROMOTION。

## 目的与对照

Frozen originalF0 identity/defaulthead plus E51 augmented rejection. Source L density/anchors and knownV calibration, independent condition U bias/discovery/anonymous registry, then frozen four-state futureQ and truth-last U-only permutation scoring.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

源统计、C0/P1 四态预测和独立 U/Q truth-last 评分共五个 stage 全部完成。主状态两条件均 NO_PROMOTION；详细性能、分组、资源观察边界见本文末尾实测结果和同名 results 文档。

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

## 2026-10-29 冻结身份的无标签适应与匿名注册实测

`20261029-phase3-unlabeled-registration-s2026092701-r01` 已完成全部 5 jobs 与独立 U/Q truth-last 评分，两个条件均 NO_PROMOTION；总目标 **ACTIVE**。身份 F0/E50、辅助 E51、G 与 source augmented 读出全冻结，无 target 标签输入或新神经更新。真实 U384 先适应/聚类/冻结，再预测未来 Q864；C0/P1 共用 1248 个物理 ID，正式 RX24-18 未访问。

| 条件 | 状态 | 旧类接收% | 旧类 open% | F0 损失 pp | REG0 未知拒识% | 新类未来 macro% | H% | 注册类 FRR% | 后续拒识% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | DA0_REG0 | 90.4514 | 71.7014 | 7.9861 | 96.8750 | N/A | N/A | N/A | 93.7500 |
| C0 | DA1_REG0 | 90.4514 | 71.1806 | 8.5069 | 97.5694 | N/A | N/A | N/A | 93.7500 |
| C0 | DA0_REG1 | 90.6250 | 71.7014 | 7.9861 | N/A | 75.5208 | 73.5616 | 12.7604 | 93.7500 |
| C0 | DA1_REG1 | 90.6250 | 71.1806 | 8.5069 | N/A | 75.5208 | 73.2865 | 12.8906 | 93.7500 |
| P1 | DA0_REG0 | 92.7083 | 69.0972 | 3.6458 | 40.9722 | N/A | N/A | N/A | 37.5000 |
| P1 | DA1_REG0 | 93.2292 | 69.2708 | 3.4722 | 36.8056 | N/A | N/A | N/A | 35.4167 |
| P1 | DA0_REG1 | 93.7500 | 69.0972 | 3.6458 | N/A | 15.1042 | 24.7895 | 11.5885 | 36.4583 |
| P1 | DA1_REG1 | 94.0972 | 69.2708 | 3.4722 | N/A | 14.5833 | 24.0942 | 10.1562 | 34.3750 |

C0/P1 均实际应用 bias；DA1_REG0 相对 DA0_REG0 的旧 open 为 −3/576（−0.5208 pp）/+1/576（+0.1736 pp），适应总目标未通过。C0 注册两簇，DA1 一簇混入12条旧类；P1仅一混合簇（old32/new35），两真实到达类合并，未单独发现者仍在未来 macro 分母。REG1 没有增加旧类正确数，两个条件新类 macro 均低于80%，注册 FRR、later拒识及旧类绝对保护均失败。

原 F0 同输入 closed C0 79.6875%、P1 72.7431%；主状态旧 open 为71.1806%/69.2708%，总损8.5069/3.4722 pp。C0 REG0拒识97.5694%仅单项过95%；P1为36.8056%。所有 source known-only 校准30组/七方法均合法，不等于目标保证。匿名ID的U真值映射只用于独立评分，不在Q上拟合；REG0新类/H为N/A，错误匿名接受不算拒识。

全pipeline 27.578251s、source统计 10.867493s，非逐包时延。实际双F＋G＋两diagnostic读出共734671参数（主MLP-only734570），未计入非梯度几何/锚点/proto等状态。独立只捕获启动driver PID/CWD/cmdline，未抓到live GPUworker/显存峰值；终态live/owned为空、foreignstop0，不为补采重跑。详情与全部分组索引见 `docs/PAMF_PHASE3_UNLABELED_REGISTRATION_RESULTS_20261029.md`。

该开发闭环完成不构成总Phase3完成；维持clean95%/star90%、FRR5%、同输入原F0open损失2pp、未来新类macro80%与旧类保留全部目标。

下一科学假设是：冻结旧身份函数仍不足以使当前星地拒识几何可分；应区分候选集的旧类污染、两到达类合并及跨日期原型失配，再检验有约束的表征/拒识证据能否同时改善未来新类识别与后续拒识。该假设尚未实施，不回收本轮 Q 调参。
