# D35-DSWR稠密安全winner条件注册实验

## 登记

- 实验ID：`d35_dense_safe_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`LOCAL_VERIFIED_PRELAUNCH`。
- 假设：D34的主要失败来自稀疏可见性而非原型量化；把所有新类改为全winner可达，并用旧support最大残差阈值保护每个winner，可去除约`-2`的新类LOO截断，同时维持旧fit support不退化。
- 比较：Z0、D25-C0、B3、D33-FAST、D35-A/B/C；7候选×3场景×5个独立held-rank折=105行。
- 本轮K10-only；K1/K5只是接口边界，不是执行或性能证据。

## 机制与协议

完整公式和实现追踪见`analysis/d35_dense_safe_registration_traceability_20260718.md`。同一已接收LEO_weak IQ仅生成一行288维`[z160,FFT96,RF32]`拼接描述；不新增信道overlay、physical sample、support row或K。query完全关闭；clean/source样本及未授权衍生信号不可达；无角色Oracle、真实batch类数、quota、global assignment或dense query图。

D35的所有新类对每个旧winner始终有有限score；winner只索引一个由旧support构造的安全阈值。A使用单mean原型；B/C保存最多2个确定性原型，但按`(old winner,new class)`support证据只选择其中1个，因此每个query、每个新类仍只计算一次288D int8 dot；C对旧floor winner加倍不确定度buffer。所有新类原型为int8+FP32 scale/inverse norm；旧FAST score前缀不修改。这里保证的是全局可见，是否可达仍必须由physical LOO margin证明。

## 成功标准

- full-K fit旧support逐类/floor不退化；15个outer held折旧类new intrusion全部为0。
- 三场景所有新类physical LOO margin_min>0，重点检查09f8和f608。
- 联合指标达到B3与D33-FAST门槛，且不转移旧floor损失。
- 0 optimizer step、active<=50k、状态<=50kB；相对identity qKNN、B3、D33报告MAC/延迟/状态Pareto。
- 即使D35注册成功，FAST注册前旧类82.22%仍低于正式92%目标；不得把Stage2-C成功描述为最终路线完成。

## 执行计划

复用D34同一密封support与receiver/seed/scenario，不新增数据准备。完成core、runner、launcher和测试后先Git提交，再执行N607直接preflight、live inventory、最小同步、SHA闭合和唯一输出检查；计划GPU0，唯一输出`runs/d35_dense_safe_20260718/output/support_screen_v1`。完成后回填105行、逐类/场景矩阵、old intrusion、新类LOO、完整日志、资源审计、RECEIPT和Git提交。

## 本地实现与验证

- 路线锁提交`005819f0`，winner条件selector修订`28264f03`；core+单测提交`2112a855`；runner+launcher+集成测试提交`7b48b223`。
- D35 core SHA`6a96b6641d40930a867d4b99fb335575daf7a47262e37f7578210e5e25c62c0c`；共享runner SHA`063dfcf6ea0182af825b4e5850a0e01d20cff9b61497a3f6948edb9498cf9c13`；launcher SHA`f2546a62b3aa0f5c06a56a827b5e631160d57fb29905491e53be459f1341ac1b`。launcher中的runner/core锁与实算一致，无占位符。
- D35、D34、D33、Fisher和共享CLI相邻回归77/77通过；core/runner`py_compile`、launcher`bash -n`、`git diff --check`通过。
- candidate set精确7候选×3场景×5折=105行。positive route除总体old/new/H/forgetting/joint floor外，还必须逐旧类、逐新类达到B3和D33-FAST两者中较高的均值阈值；09f8、f608、14-7和其他floor不能被总体均值掩盖。
- runner将全局可见与LOO可达分开；full-K10门继续要求outer held旧侵入为0、全部新类LOO转正、fit旧类逐类/floor不退化、资源/协议闭合。`selected_positive_route`来自full-K10后的最终decision。

20新类K10合成扩展回归仅验证资源公式：A/B/C的D35增量均为5,760MAC/query，与FAST合计7,776MAC，相对identity-only K10单qKNN的41,600下降81.31%；组合状态A为14,644B，B/C为20,584B；部署refit约1.602M/3.101M MAC，0 optimizer step。合成old LOO仍出现33/34/33次侵入，证明硬门真实生效；该回归不是性能证据。

当前D35只把目标新类原型压为int8；Stage2-B仍使用FAST旧头，没有声称目标旧support int8原型路线已经闭合。若D35注册层通过，下一轮必须加入目标旧类int8原型与授权Phase1 int8锚的轻量融合，专门提升注册前旧类与floor；若D35注册层失败，则先修注册再叠加旧头，不能用未实现的旧int8原型作成功叙事。

## N607计划

- 最小同步仅3个文件：`code/cvsrffi/stage2_d35_dense_safe_registration.py`、`code/scripts/run_d25_support_only_concat.py`、`code/scripts/launch_d35_dense_safe_20260718.sh`；不上传本地有他人改动的`stage2_diag_cosine_exploration.py`，远端必须继续核验固定SHA`14ec9193...1ca`。
- 计划cwd`/home/szu2070436088/2510044040/CV-SincNet`；Python`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；命令`D35_GPU=0 bash code/scripts/launch_d35_dense_safe_20260718.sh`；日志`logs/d35_dense_safe_20260718/support_screen_v1.log`。

## N607启动前闭环

- 09:30 CST直接preflight与live inventory通过：8张RTX 3090均0%/10MiB，训练进程0；目标输出不存在；`/home`可用7.6TB；SSH/TCP22均已退出。
- 只同步core、runner、launcher到约定同路径。远端实算SHA与本地完全一致：runner`063dfcf6...f9c13`、core`6a96b664...62c0c`、launcher`f2546a62...ac1b`；未同步diag仍为固定`14ec9193...1ca`，D33/Fisher依赖也闭合。
- 远端launcher`bash -n`、core/runner`py_compile`通过，唯一输出仍不存在。GPU0允许新增本任务1个进程；启动后回填PID、运行状态和最终artifact。
