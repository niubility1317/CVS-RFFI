# 固定三成员概率集成：零训练技术恢复

原 r01 全部 6 个新成员完成固定 30 轮。聚合 source_contract 写错为参考配方，缺少实际加载器产生的 native_role_comparison，原校验正常拒绝。独立读取确认所有原 owner 进程退出，未产生任何 prediction 或 scores。保留原权重、日志、失败产物；本恢复只在新 run/output 写元数据和预测，0 新训练、0 新更新。

只接受原聚合文件恰好等于原参考配方的已知缺陷。分别核对两个真实子成员运行时契约与原参考配方，包括实际物理角色及 EXACT_MATCH；从实际证据重建新目录的聚合记录，不添加虚构标记、不放宽 verify_contract。原 9 个 checkpoint 的配置、实际初始化种子、固定30轮更新预算、冻结来源和继承路径校验完整保留。原代码同时修复写入逻辑，旧 release 不修改。

独立 P0/P1 定点复核 PASS：真实子契约逐个验证；只允许已知精确参考副本；所有原加载检查保留；新目录不写旧产物，0训练。聚焦负测覆盖缺实际角色标记、错物理角色、变更配置；合成7视图/3种子闭环验证缺视图时truth不打开，378指标记录及3528000预测。一次真实原checkpoint no-query smoke在N607 preflight执行，PASS立即自动接续，无额外审批或hash门槛。

沿用原源资格、相同三primary种子2026092701/02/03，每seed3个独立随机初始化模型，固定softmax温度1等权平均。不会依据目标成绩更换成员、拟合融合权重或重跑。每row冻结后立即自动clean及六practical，每视图168000预测；全部7视图完成后独立进程truth-last评分。完整后报告三个seed及各RX/day/TX/场景，主判定clean三seed均值>=84%。此基准已暴露，不称新盲测。

成本沿用原总9成员59940更新/270模型轮次，每ensemble531075参数，为原单成员3倍训练和推理。本恢复仅新增预测/评分时间。唯一owner，新不可覆盖run/release/output，普通账户；每GPU最多3个总实验进程含其他owner，容量按现存进程核对，不停止其他健康任务。仅预登记系统技术失败、权限或输出碰撞终止所属run，不按准确率停止。矩阵/命令/CWD/环境/路径/GPU规则/stoprule见experiment.json。一次归档本地远端SHA比对、remote compile、preflight、PID/CWD/cmdline/GPU/log增长核查后自动完成。
