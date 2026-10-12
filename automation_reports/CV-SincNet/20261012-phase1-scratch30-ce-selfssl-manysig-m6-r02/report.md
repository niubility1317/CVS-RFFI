# 从头30轮：纯CE与可更新伪标签对照预登记

用户明确新预算：“从头训练30轮”。本批取代后续matched controls的E20预算，保留既有run，不热改、停止或重启。clean三种子均值84%仍为目标。

## 吸收前对话与优化报告

旧响应监督RESP_LU80.3940%低于充分匹配额外clean80.5415%，尚无独立身份收益；完整源诊断不支持“读出只依赖q”，但旧目标兼容不等于当前真实等变性。新波形不变量源先导未通过；有限实际间隔+0.4370个百分点且2/5RX提升、软度量+1.1630个百分点且3/5提升均未过源条件。普通cos原型2701虽+4.3481个百分点，独立2702为+2.3037、2703为-0.8556，稳定条件未过。所有负结果保留，不降低门槛，不把它们拼入新模型。

按报告先建立真正有效的训练控制。两个方案各3seed2026092701/02/03，固定从零30epoch×222=6660updates，复用原L6300/U56700/V27000物理角色与source/target RX互斥，无旧checkpoint或teacher初始化。原时频/PA/参考响应、内部身份门控与cos分类头保留，显式CFO槽仍屏蔽；这不保证rawIQ完全不含频偏。

## 固定实际配方

CE30：L clean CE，label_smoothing0。U缓存/迭代/教师/伪标签/增强/辅助前向均实际关闭。

SELF_SSL30：前5epoch/1110updates仅L；其后每update从独立U遍历读取128包。当前student以clean eval/no_grad产生新的软目标，固定95%confidence+同opaque PID两次相邻完整遍历同预测门控，不读取U真TX。强view为固定std.015数字噪声，student train前向计算全batch分母maskKL；5epoch/1110update线性ramp至weight.25。没有固定旧teacher、EMA、域/物理/关系网络或星地训练增强。U噪声及dropout使用独立每步RNG，不消耗下一步L随机轨迹。U目标随student更新，不是永远复制原E20。

两方案均AdamW peak2e-4→min1e-6，30epoch cosine，wd1e-4、clip10；优化器及所有身份网络从零，不恢复旧轨迹。L同seed同RX/day全遍历序列；SSL增加U计算，明确比较相同epochs/主L预算，不冒称相同FLOPs。V仅epoch10/20/30健康评估，禁止更新和选模；每行只最后30轮checkpoint冻结。

## 自动测试与科学判定

每行冻结后立即进入预测，不等整批：clean+六practical全部168000物理记录，无重生成观测、query拟合或归一化更新。所有该row七视图prediction固定并核对后，exec到独立scorer才打开truth；每row126个RX/day/TX/整体指标，两实现混淆复算。六行共7056000预测，完整三seed平均/SD、配对差、worstRX及混淆保留。主SELF_SSL30 clean均值至少84%才记目标达成；其他视图结果完整报告，不能用sourceV或诊断替代。

该基准已反复暴露，报告如实标明开发比较，不声称全新盲测或真实卫星在轨性能。逐row目标结果不改变其他训练、参数、排程或seed重跑。没有候选筛选或按目标选择行。

## 发布

矩阵、commit、精确命令/环境/CWD/路径/GPU/owner/产物在experiment.json。一次源码commit、聚焦负测/合成七视图闭合、独立P0/P1、真实checkpoint无querysmoke、一个archive比对与remotecompile后继续。reuseVALIDATED_ONCE数据，不重建/重hash成员，不要求报告seal。每GPU最多3个总CUDA进程。只允许权限/错误角色/输出碰撞/无法执行或收尾闭合等系统技术停止，低性能不中断，不动其他健康run，保留全部产物。

## r02 技术恢复

r01 的CPU真实预检PASS，但调度器在逐row配置核对时退出。独立startup日志、PID退出与run-root仅含空logs证实0worker、0训练、0预测。原因是Windows预登记U_cache用str(Path)，Linux用POSIX。回归在修复前以PRE_REG_CROSS_PLATFORM_U_PATH_MISMATCH失败，改as_posix后通过；唯一变化是路径序列化及新不可覆盖run/release/owner，矩阵、30epoch、seed、算法、数据和scorer保持。原归档与失败root保留，无进程终止或重启健康任务。
