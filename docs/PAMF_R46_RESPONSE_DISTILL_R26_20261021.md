# R26：等额外前向成本的扰动输出与响应蒸馏

状态：VERIFIED_QUEUED_AFTER_REAL_SMOKE。已发布并通过真实源 IQ 检查，等待 R23 对应 GPU 进程完成；未启动新版 GPU 训练，无新版正式分数。

目的为在 35.05 万指定父版基础上联合压缩和提升身份识别性能。R25 仅源 IQ 诊断显示，较低输出误差并不保证保留相同局部响应；带信道源响应又不能唯一解释正式 TX1 退化。故本轮预先固定两个同结构、同预算、同额外完整 G 前向次数的新训练对照，检验训练目标差异，不把 R25 指标当性能门槛。

两版均仅从用户指定原 R46 prune_refine350521 source-final E20/4940、方法 bbb8daf817de050b958d217b70d4c33f4aefe341 初始化并用作固定 IQ 教师。独立 data-free SVD625 得到完整 278180 参数（coarse106658/action80045/prior91477），保留全部通道、GN4、skip、3stage/6CG，所有组件联合更新；不加载任何 R18/R22/R23 学生权重。训练基础为 paired clean IQ+.05 powerSTFT64/128/256+stage教师IQ线性1至.1+.02分组复FFT+.25clean恒等，沿用源合法完整祖先和原7/63/30物理角色。

| 固定版本 | 完整参数 | 新增损失权重 | 每次更新额外 teacher/student G 调用 | GPU |
| --- | ---: | ---: | --- | ---: |
| pair_aug_control | 278180 | .01 | 正负扰动各一次，各模型共2次 | 2 |
| odd_response | 278180 | .01 | 正负扰动各一次，各模型共2次 | 3 |

每步取原源随机排列批次的前16条 IQ（tail batch24也取16），不按 L/U/TX/RX/标签或指标选择。扰动周期按 update 固定 random0.005、phase0.005、random0.010、phase0.010，19700更新每格4925次、78800条源探针。random 使用独立 CPU generator seed2026102126+update，复 RMS 归一到当前 received，phase 为 i*x；不改变全局训练随机状态。四个额外全 G 输入为同一 x±delta，两行输入和教师前向次数相同。

pair_aug_control 对两个扰动后的教师输出分别作相对 IQ 误差并平均，分母为各自 detached 教师输出复功率clamp1e-8。odd_response 比较 (Splus-Sminus)/2 与 (Tplus-Tminus)/2，分母 max(教师odd功率,.01delta功率,1e-8)。教师在 no_grad 下 eval，所有参考 detached；学生正负两支都保留梯度。基础中心 IQ 目标继续约束中心输出；有限响应约束本身不能证明身份保真，也不是精确 Jacobian。系数 .01 在任何本轮新目标访问前固定，不作目标驱动搜索。

每版 E20/19700、B64micro64；每轮原6300Ls/56700Us全曝光、received01至20，不重划分或新模拟。累计L126000/U1134000；基础received学生/教师各1260000，clean学生1260000；探针选中315200条，额外teacher/student各630400输入，教师总1890400、学生总3150400。记录探针实际L/U和四格计数，均不读取TX真值。两行额外完整前向次数相同，实际壁钟/峰值内存和损失计算成本分别报告，不声称壁钟完全匹配。

ResetAdamW2e-4、warm985、cos至2e-5、WD1e-4、clip1、FP32/TF32off，所有组件累计100轮、组件和300、祖先串行120+新增20=140。固定最终E20/19700；源1800×7、同卡完整G/G+F成本和独立源评分后，逐版自动正式168000×clean/三郊区/三urban并独立truth-last。源负结果仍完成正式矩阵，七预测全部固定后才连接truth。原F50 seed2026092701及默认头只在冻结评测与无query smoke使用，无新头或F训练知识。

逐景accuracy/MacroF1、TX/RX/day、混淆、repair/damage、clean损伤、波形/FFT/F几何、完整训练曲线、全coarse/action/prior/物理求解成本均保留。父逐景0.8pp/1344条、原R17历史1pp是退化容差；lowurban70%、其他五景距rawclean5pp、clean保持及强TX真实性继续单列。参数和ConvLinearMAC减少不代替实测延迟；同卡B1/B32三轮正反顺序覆盖完整G/G+F，计时不含传输/前处理，MAC不含FFT/CG/norm/访存/F。

N607普通账户，GPU2/3最多2own、每卡1；等待 R23 同卡已登记实际PID/完整命令退出及终态，再经共享锁 free≥18GB/allowner<2 分配。忙时等待，保留所有健康owner。CPU评分2线程、R06score.lock、最多1ownscorer。一个方法commit、新不可覆盖run/release/output、一次preflight、归档两端一次SHA、一次compile；launcher首步真实原父/F50/2Ls smoke含新增探针反传与严格重载，PASS立即继续。技术失败保留影响行，健康行继续，无自动重启/热改/回退。精确配置、环境/CWD、命令、路径、种子和矩阵见experiment.json。

37项检查覆盖新数学目标、梯度、教师状态、相同输入/曝光、零能量、全模型严格重载、默认七景链、truth-last与旧输出保护；唯一P0/P1审查PASS，无P0/P1。通用独立设计审批/全库测试额外门槛记REJECTED_EXTRA_GATE/NONBLOCKING。历史目标已暴露，按工程对比披露；新矩阵不按目标成绩调参、选模、改教师或选择性重跑。主对话检验中间特征约束，本组检验有限输入输出响应约束，部署结构不增加。


## 实际发布与排队核实

方法 commit 781e56d9d2bd401d56ad7b758ed4f4812c04f1e4 已推送并独立匹配远端 OID。一次 N607 预检确认原父 source-final20/4940、源 READY/EXACT_MATCH 和正式七景最小句柄；只读元数据，未读目标 IQ/truth。归档两端一次 SHA 与远端 compile 通过，实际 release/CWD 与登记 r26 路径匹配。

原父及原 F50/默认头、真实2L_s无query smoke PASS；两行新增探针各执行4条额外教师输入和4条额外学生输入，全部coarse/action/三prior梯度有限非零、原父和F无梯度/无状态更新，完整学生权重严格重载输出一致。新增probe实际random0.005，两个真实源样本均执行正负全G路径。

实际CPU控制器PID/CWD/完整命令/方法commit匹配，日志持续增长，无failure/Traceback。当前两行均等待已登记R23同卡owned进程终态及退出，R26 GPU worker数为0。预检时GPU2/3各已有2条所有owner进程，不申请额外GPU；容量在原句柄退出后再查。R23最近完整可用源日志为15/20、8/20，23条记录均已核对，F/TX/V/target读取与训练改动均0；这不是正式性能结论。

远端每行训练完成后自动执行源7、完整成本、独立源评分、正式168000×7和独立truth-last，无源收益门槛。最终本地汇总/权重交付工具继续按后续完成阶段补齐；当前不把排队状态写作训练或正式测试完成。目标保持ACTIVE。
