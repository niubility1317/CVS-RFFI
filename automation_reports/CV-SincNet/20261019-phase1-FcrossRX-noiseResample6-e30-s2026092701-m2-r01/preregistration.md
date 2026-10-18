# P75：原两次 As CE 中仅重采样新增噪声

状态 PREREGISTERED；尚未训练或评分。最近P74冻结源归因发现：固定接收状态去新增噪声可提高mid/low，但没有单个整贡献满足四条件局部解释；删除PA/REF/SIG反而损伤urban。P70/P71风险、P72mask、P73概率CE固定候选均关闭。另一对话负责U路由与共享/分离AdamW，本轮不重复。

唯一假设：同一完整非噪声传播条件下，现有第二As前向使用独立新增噪声，能否比重复同一受损IQ产生更有效的身份监督。不是新损失或新噪声模型。旧heldRX1十L INDEP/ANTI做过双噪声监督，但没有SAME_NOISE控制；ANTI负结果不证明此隔离比较无效。P68重抽整个信道与场景，本轮只换一个native noise流，这一关键假设不同。

两新行C01 SAME_NOISE_CE6与N01 INDEP_NOISE_CE6，均从同一已核实SAT18 AS_CE scratch20 student/ownEMA独立追加10（新AdamW/cos10/CENTER_UNITS），累计E30/6660更新，各新2220。不加载原P43 As20、E50或按目标选择的父。共同父前20轮As KL已经关闭，必须披露该历史，不将整配方相对P43As30变化全归给尾段噪声。原P43 NoAs30/As30与P68独立完整信道CE6作为三个描述参照，复用固定源/正式预测不重跑。

保持完整L6300/U56700隐藏/V27000、RX1/3/4/6/8/day1/2/3、split392005、6TX、无heldRX/Ldev。F+CROSS_RX/SIG_TIME_X3/统一Sinc/NoExplicitCFO/joint160/defaultcos30随原流程联合训练；不新增头、分支、损失、教师或U门控。IQ2×256/fs25MHz，原RMScenter=false；As直接原raw，不预叠ordinary。训练fc2.462GHz、正式fc2.45GHz不改。

六L前向和总有标签CE系数1.25保持，As两CE各.125、总.25，As无跨视图KL，ordinary/raw保留原RDrop和SupCon，全部合法L增强均接受真实标签监督。C第二slot复用原AsIQ；N第二实例用第一cfg/physicsseed/exact RX硬件对象，只替换latent.rngs['noise']为新命名Generator，原process保留alpha/beta IQ噪声变换和G。EQoff G1不代表噪声一定proper；不能直接输出端加白噪声或精确校准噪声能量。

锁定/质量/残余scale/频率/phase/IQ/EQ流按原非噪声实现重放；AGC依赖实际signal+noise，各view按原native重算，**不冻结首AGC**。第一namespace与物理种子不变；noise1 suffix=/view1-noise-independent，每epoch/step/physicalpacket刷新。session按RX/day/eq而非TX。训练、源面板和正式流分离。

两个源条件在新正式分数读取前固定：N的两draw×mid/low均值相对max(newC,原P43As30)至少+0.5pp；完整源V clean N相对P43NoAs不低于−0.1pp。仅两项source比较，不新增特征距离/伪标签代理门槛；失败关闭固定候选，不扫描噪声强度/权重/seed/E50。所有新行均完成clean+六practical固定预测，逐row全部prediction固定后独立exec scorer连接truth。源资格只决定未来预算，不证明正式未知RX保持；不以正式分数调整或选择性重跑。

一次Git代码/config身份提交待发布写入；CWD PROJECT/releases/cvs_FcrossRX_noise_resample6_E30_20261019_r01，命令 CVS-RFFI/python -u experiments/cvs_noise_resample75/runner.py；输出唯一 PROJECT/runs/20261019-phase1-FcrossRX-noiseResample6-e30-s2026092701-m2-r01，logs逐row独占。N607既有普通账号/Python3.10 Torch2.1 NumPy2.2，FP32 CPU2，每GPU启动时含所有owner最多2实验进程、free≥12000MB。一次preflight、归档传输一次SHA比对、远端compile、唯一P0/P1检查、首次真实CPsmoke通过即继续；不修改健康任务。

新增20model-epochs/4440更新；每row native3162962/source12600/formal1176002，shared smoke1540（studentownEMA4+两真实sixL合成1536），共8,704,668新增整网样本。额外模型NN0、教师/头拟合0；C native快照142080，N284160，sharedsmoke192，总426432；unique非noise latent两row284160+smoke128。复制的第二slot不算新信道；生成成本和墙钟实测，不声称物理重采样免费。单个共享AS_CE20父成本单列，不重复算成两份新训练；两条逻辑轨迹各总E30。

停止只针对run-owned角色/来源/实际native噪声链/非有限/预算/输出碰撞等技术错误；保留失败产物，不自动重试，健康peer完成。科学负结果不停止已授权评分。

预期产物由experiment.json列出：实际模块__file__/resolvedargs/来源、完整epoch/step/physical记录、final权重与sourcefreeze、source决定、all7预测和独立评分、五logicalrow分析、训练曲线/状态质量分层/pairederrors/真实成本图表。一个modelseed，两source draw不算跨seed。最终仍需回答正式星地增益、clean代价、是否满足.1pp与是否有可重复双提升，不把单seed微涨或源oracle当成功。
