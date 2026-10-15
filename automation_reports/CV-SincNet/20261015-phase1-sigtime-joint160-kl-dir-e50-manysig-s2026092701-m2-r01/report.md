# 固定 SIG_TIME：KL 与实际 joint160 方向迁移

状态：LOCAL_VERIFIED，尚未发布。用户 2026-10-15 附件的第一步：实际 joint160 非对称方向蒸馏。复用已经固定的 E50 C3 为 A；B=OUTPUT_KL、C=JOINT_DIR 各从零训练 E50，model seed2026092701，33 轮 label＋17 轮旧门控/ownEMA，222 steps/epoch、11100 次更新。唯一矩阵保存在 experiment.json；本轮不扩 seed。

SIG_TIME、旧内部门控、参考响应结构和默认 cosine6×160/scale30 保持，全部仍由原 CE 联合训练。无新分支、分类头、预测MLP、独立拟合诊断头、冻结骨干再训头或新部署模块。CE clean:satellite=1:1，各自均值，无学生正确/置信门控。L6300/U56700/V27000、源RX1/3/4/6/8/day1/2/3、split/data392005；U 真值隐藏，原 U 旧门控/EMA监督保持。

两条新候选均用同一样本 ownEMA clean eval 的 stop-gradient 目标。仅 L_s clean 教师正确且属于 urban 场景时参与额外项，全部合法有标签 satellite 始终参与 CE。首轮不新增 urban U 特征目标，避免把额外监督量和一致性对象混在一起。B 用默认头 W.detach 的输出 KL；C 直接对实际部署 joint160 的 eps1e-4 单位方向计算1-cos。两者每次 satellite 前向都验证辅助重组特征与原 joint160 逐值相同；只 detach 融合 gain，不 detach 分支。辅助 W/gain 梯度为0，原 CE 照常更新它们。

系数只由独立 source-only 零更新校准取得：run 20261015-phase1-sigtime-joint160-calibration-manysig-s2026092701-r01，code5b2d53e6db5cd8c6207af1709a0dbe4771d7e1b2。合法 L720，mid/low urban、2250 前向、0 更新、6.498 s；源 C3 与自有 EMA 状态不变，U/query 未读取。KL lambda0.04998551146175206，方向lambda0.44114306609656917，分别使五组/源batch的加权梯度比中位数为0.1，无截断。source_recipe.json 在新训练和读取 C3 目标成绩前已固定。共同 eta 是 A 已实现的 source-only 50 轮日程，两条新训练均只重放、不按自身结果推进；物理增强与 A 在相同命名随机流逐值一致。A 权重只供校准和 smoke，不用于初始化、resume、teacher或EMA继承。

两条新行网络/数据/增强/EMA/预算/额外前向匹配，只改变辅助目标，因此 B 对 C 是匹配比较。A 的既有间隔使用学生 clean 参考和接收质量权重，而 B/C 改用 clean EMA 正确资格；A 也少一条 L clean 教师前向。A 对 B/C 是完整迁移配方比较，不宣称单独归因于某一个公式。所有差别如实报告，不把旧A补训或选择性重跑。

训练只用 practical_low_suburban、practical_mid_urban、practical_low_urban。实际导入 experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py，25MHz、IQ256、原 native RMS 后模拟器输入 RMS；residual/post_sync/EQfalse（G=1）、reference_snr_db30、quality adaptation/phase tracking/IQcomp true、reflect；输出不再 RMS。几何/传播/遮挡/噪声/接收状态/补偿依赖、RX/day session稳定与跨epoch/view刷新保持。正式 IQ 和信道完全不改。

额外 L clean EMA 前向每行1420800，L clean和satellite各1420800，原 U clean教师/strong学生各963900；每行主训练总前向预期6190200，另列50×720×4源面板、原 V 验证及 smoke。实际墙钟、前向、更新、裁剪率由全量日志核实，不预先承诺速度或收敛。t1/t2/SIG local/PA投影/ref投影记录加权辅助/.5satCE梯度比和夹角、辅助W/gain零梯度；每epoch记录实际联合参数更新、gain、分支方差和TX×RX×状态覆盖。主训练 Dropout 完全沿原路径，不能把教师eval/学生train差异全归为信道损伤。

正式评价每行固定final50后即启动：clean＋practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban，各168000条固定received IQ。同row七种prediction全部完成后，独立exec scorer才连接truth，自动保存RX/day/TX/overall及Macro-F1，并独立复算。报告 low/mid urban、urban均值、最差RX、clean/suburban及负差。目标60%、相对普通拼接urban+3pp/low+5pp/clean降<=.5pp均为验收探索目标，不回流调参、选模、晋级、重跑或自动扩种子。一个seed不构成跨seed证据；历史benchmark已暴露，不声称新独立盲测。

本地命令：E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_identity50.publish --output E:/type10-7/local_artifacts/cvs_joint160_identity_20261015/training_delivery_r01。远端命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_urban_identity50.dispatch。CWD PROJECT/releases/cvs_sigtime_joint160_kl_dir_e50_20261015_r01；run/output PROJECT/runs/20261015-phase1-sigtime-joint160-kl-dir-e50-manysig-s2026092701-m2-r01。普通账户，owner codex/root/joint160-kl-dir-e50-20261015；torch/OMP/MKL/OpenBLAS2线程、FP32原精度设置，GPU共享锁动态分配，总进程<=3/GPU且计入其他owner、空闲>=12000MB。实际代码/config以release_commit.txt的Git commit标识。已有健康E50/其他任务保持。

最小发布：一次代码配置commit、11项定点检查、一次限定P0/P1审查、登记、一份archive传输比较、一次远端编译预检。launcher首步读取真实合规C3 checkpoint、验证实际joint160及source_recipe后立即继续从零训练；不接触query。每行训练最初也保存并重载自身scratch checkpoint smoke。随后独立核实PID/CWD/argv/GPU/日志增长。角色/来源/配置/预算/覆盖/非有限/输出冲突等系统技术失败保留产物并停止pending行，允许健康active行完成；不自动重训，不杀其他任务。性能不达标不是技术重启理由。

本地实际验证：`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_identity50.checks`，11项PASS，含两种实际新Observer/label阶段无U metadata调用、同流增强逐值相同、无额外参数、teacher停止梯度、CE继续更新原头/gain、辅助W/gain边界、错教师/隐藏标签拒绝、预测未完成禁止评分；stderr保存在local_artifacts/cvs_joint160_identity_20261015/training_checks_stderr.txt。源校准已独立完成并验证。新的训练接线唯一一次独立P0/P1审查完成，指定范围未发现P0/P1；默认头联合CE、ownEMA教学、辅助梯度边界、E50预算/源配方和七视图独立评分接线一致。不重复审查上一阶段未改的校准实现。

设计对应记录：analysis/cvs_joint160_distill_traceability_20261015.md。本轮是第一阶段实现：容差目标、同信道高质量中间视图、分组SIG_TIME局部教学、PA/ref局部目标及额外U特征目标均未运行；不以当前实现冒充完整多阶段设计。冻结后的正确/错误TX特征距离、固定clean正确集合修复与新错、塌缩/gain诊断另行按同一合法源面板执行，不额外拟合头。

实际发布状态：RUNNING / VERIFIED。code/config commit `704a3e917de3bd8f8a739fddb2a21212b2840148` 已push，独立远端branch OID一致。普通账户N607预检、一次archive传输比对、远端编译PASS；launcher真实合规C3 source checkpoint/no-query smoke通过后自动启动两条scratch训练。

OUTPUT_KL-s2026092701：GPU1、PID2898654，CWD/argv正确，日志增长5170字节，实际完成2轮。首轮所有五个特征组、默认头与三组gain均有实际联合参数更新；新增aux的W/gain梯度全部为0。L/satellite/ownEMA前向计数见launch_verified.json，未读U真值/目标成绩。

JOINT_DIR-s2026092701：GPU2、PID2898731，CWD/argv正确，日志增长5171字节，实际完成2轮。首轮所有五个特征组、默认头与三组gain均有实际联合参数更新；新增aux的W/gain梯度全部为0。L/satellite/ownEMA前向计数见launch_verified.json，未读U真值/目标成绩。

核实证据：E:/type10-7/local_artifacts/cvs_joint160_identity_20261015/launch_verified.json；monitor/latest.json持续只读观察。初始reference gain=0时投影梯度为0是原初始化行为；CE先更新gain，其后真实投影梯度已记录，不改变原结构/初始化。源校准0.1是校准面板的中位比，训练过程不强迫达到0.1、不在线调lambda。
