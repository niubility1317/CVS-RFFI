# IB-RFF 作者四折补全预登记

首折 run `20261019-phase1-ibrff-author-manysig-s2023-r02` 已完成并保留，不重训。补齐作者已固定的其余三折，以四折完整结果判断能否接近作者代码日志；不因某折目标分数低而修改参数、换 seed、筛折或重跑。原论文全文尚未核准，结果不等于论文表验收。

| run_id | 作者折 | 目标 RX | alpha | 作者日志准确率 |
|---|---:|---|---:|---:|
| 20261019-phase1-ibrff-author-manysig-f1-s2023-r01 | 1 | 2-19、3-19、7-7 | 0.001 | 88.53% |
| 20261019-phase1-ibrff-author-manysig-f2-s2023-r01 | 2 | 7-14、8-8、14-7 | 0.08 | 96.28% |
| 20261019-phase1-ibrff-author-manysig-f3-s2023-r01 | 3 | 18-2、19-2、20-1 | 0.1 | 93.58% |

每折源 RX 为其余 9 个，源 day1/2，目标 day3/4。复用既有全 12 RX、4 天、6 TX、前 100 包非均衡数据导出，不复用任何其他折权重。每折 source7560/3240、target3600，seed2023。折分顺序与作者 `split_receivers` 相同；alpha 固定取公开对应日志文件，选择来源未核准，不声称新增独立盲测。

各折独立从零进行两阶段训练，每阶段最多 200 轮；阶段一 alpha0，阶段二使用表中 alpha。其余模型、均功率归一化、Adam0.001/batch128/weight_decay0、源 NLL 选模、plateau 和早停及原分类头均沿用首折。科学配置为 `paper_reproduction/ibrff_20261019/candidate_round{1,2,3}.json`。源/目标物理角色由固定折分生成，checkpoint 严格核对本折 source 契约；禁止跨折继承。

固定四折均值参照为 93.57%；沿用首折允许低 1 个百分点的工程口径，均值下限为 92.57%。同时完整报告各折差值、Macro-F1、各 RX/day 和实际轮数/成本；阈值只作报告解释，不用于调参或重跑。只有四折评分均完成才报告完整矩阵。

唯一 launch owner 为 root-ibrff-20261019；N607 普通账户 SDG-SEI 环境（Python3.8、torch1.11、NumPy1.24），首选 GPU2/3/4 各一行，不干预其他 owner，每 GPU 总进程不超过 3。各 run 独占全新 release/run/log 根；精确路径、命令、GPU 和代码 commit 存在对应 `experiment.json` 所引用的私有 `runtime.json`。命令模板：`SDG-SEI/python -u RELEASE/pipeline.py launch --runtime RELEASE/runtime.json`，CWD 为本行 release。

每行第一步实际 source checkpoint 无 query 烟测，PASS 后立即继续正式从零训练；两阶段完成冻结后再自动导出 query，预测器不读取 truth，全部预测固定后独立 scorer 评分。停止仅限本 run 的错误契约、输出碰撞、不可执行、非有限源 loss 或无法生成合法预测/错误真值连接。保留失败与部分产物，低分不触发技术停止。

发布沿用既有候选独立 P0/P1 审查，无重复全量审查；折分改动用 source 路径 spy 核对四折顺序、互斥及每个目标 RX 恰好覆盖一次，round0 契约保持兼容。发布前完成代码 commit、一次 preflight、一次 archive 传输 SHA 对比和远端编译；发布后核对 PID/CWD/命令/GPU/日志增长。预期每行保留 smoke、stage1/2 checkpoint 和完整 epoch/config、冻结预测、评分与 completion；真实数据、权重、预测及原始日志不进 Git。
