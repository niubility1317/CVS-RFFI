# E30 固定适配器源域诊断预登记

本诊断只使用既有固定终轮模型与原契约有标签源 L。固定 180 个物理样本来自原 observedChannel 面板，每个 TX×RX×day 单元两个样本，无性能筛选。U 的 TX 不可见，V 与 target 均不新增前向。

唯一问题：局部 adapter 的受损观测拟合不足与 clean 函数漂移各有多大；零输出 adapter 的随机 Dropout 与 eval 教师差异是否实际存在。该差异本身不证明它导致 E30 退化。旧 S3 的 SOURCE_FAIL 决定保持，不延长或扫权重。

逐行矩阵与精确命令见 experiment.json。八个冻结模型各读取 raw、末次实际训练 As、两次新 mid urban、两次新 low urban，共 1080 NN/模型；零输出模型另做四次 raw Dropout 读出 720 NN，一次真实 CP 单源样本 smoke 后立即继续。合计 9361 NN、900 次原物理信道生成、180 次合法 raw L 读取，新增训练更新为 0。继承训练成本分别保留 E20、累计 E30 与 Ts 历史 E200，不把读出叫新训练。

重建末次训练 As 使用实际 L 序列的 cumulative E21—E30 namespace、原 channel seed 与 RX/day/eq session，先重建相关完整128场景选择并与原日志计数核对，然后只生成所选180包。TRAIN_SCENES 是原三种训练场景，不与六种正式测试场景混用。新信道 namespace 预先固定为 FdualAdapter/sourceFit/20261019/draw0 与 draw1。此为命名随机流重建，不声称与未保存的历史 IQ 字节作过比较。

冻结默认头与实际 joint160 同次输出；报告源拟合正确率、NLL、宏 F1、TC 正确集合保持、新增错误、归一化特征距离及余量充分界。Ts 结构不同，只读 logits，原生 scale5 与学生30分别标明。Dropout 单独启用 identity 路径13个 Dropout，整体保持 eval，无梯度/优化器/EMA更新；参数和持久buffer前后逐值核对。

N607 单普通账户 worker，CPU2、FP32/TF32off；GPU4/6/7 单次分配，共享锁下 all-owner≤2 且 free≥12000MB。一个新不可变 run/output/release，遇数据角色、来源、非有限值、预算或输出冲突技术失败即保留产物，不自动重启、不动健康任务。

代码/config 使用本次 Git commit；一个发布包仅做一次本地/远端 SHA 比较与远端编译。先完成窄负测试和一次 P0/P1 审查；源 smoke 是执行第一步，PASS 后直接继续，无附加审批。

结果不能证明未见 RX 泛化、反推物理上限或构成模型多种子结论。已有完整正式预测不重跑，本 run 无 target scorer。

实际诊断代码 commit：`a2c9694f97e4d314207795ba774ea348cfe8c0fa`。发布归档保留含此代码的当前 Git HEAD，`release_commit.txt`记录归档身份。

## 实际状态

SOURCE_ONLY_ZERO_UPDATE_COMPLETE / ANALYZED。完整结论见[final_analysis.md](final_analysis.md)。已核实9361身份前向、900信道生成、180合法L物理样本、0训练更新/新增V/target前向。首个进程读回时已结束，保留该证据限制。固定S3配方保持关闭；raw补充路径只完成维度与成本可行性核对，尚未实现或启动。
