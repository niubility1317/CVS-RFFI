# A1当前候选延长训练与20轮测试

## 授权与选择

用户要求“设计并启动几个实验，就现在最好的方法版本，延长其epoch，也再中后期进行测试，每20轮一次”。本轮只新增实验，现有健康任务继续运行。

候选依据为截至2026-09-10 09:13已完整解析的源域V结果、机制执行、耗时和稳定性。证据见`analysis/a1_status_full_20260910_0912/analysis.json`，不依据其target评分排序。

- 主候选X2_CROSS_RX_VIEWS：E200源V clean98.4741%、LEO均值93.6346%，已完成scratch候选中的LEO均值最高；累计epoch耗时11.85小时。跨接收机clean路径200轮、LEO路径121轮实际执行，有非零identity梯度。
- R3_REFERENCE_CLEAN_CROSS_RX：E200源V clean98.5741%、LEO均值93.3667%，累计epoch耗时16.22小时；保留重建结构作为较高clean表现的家族对照，不称其为目标泛化最优。
- 不是多seed优越性结论。未完成的16行机制筛选不参与本次候选排序。

## 预登记矩阵

|行|版本|轮数|固定目标测试epoch|GPU|
|---|---|---:|---|---:|
|X2_E400|X2跨接收机clean/LEO|400|200、220…400，共11次|6|
|X2_E600|同X2|600|300、320…600，共16次|3|
|R3_CLEAN_RX_E400|R3参考+clean跨接收机|400|200、220…400，共11次|0，空位自动入队|

每次完整输出clean、leo_clear_weak、leo_low_elev_weak、leo_rain_weak，共672000条预测。最终比较固定E400/E600，同row报告四场景、逐类/接收机、训练时间、显存、稳定性。目标测试仅作探索观察，固定跑完，不据其早停、改超参数、选择checkpoint或重跑，不作为独立确认或晋级证据。

延长的是完整课程预算：原200轮时钟分别按2倍、3倍拉伸，保留学习率相对形状、DAOT/R3/U-LEO课程及MUSE/RC4阶段比例，未另加连续LR或新组合机制。因此不能把新E200与旧E200解释成同一课程阶段；新E400与E600也不能作为同训练成本比较。新实验使用已修正的当前代码，其历史E200仅用于源域候选依据，不冒称逐bit复现实验。

## 输入、继承与输出

全部`from_scratch=true`、`a1_scratch_only=true`，不加载任何历史baseline、teacher、resume或预训练权重；EMA仅由本run随机初始化student产生。不存在待审核的继承checkpoint。正式启动核对argv、初始化日志；训练完成再由自身checkpoint元数据验证来源。

保持既有ManySig equalized、seed392005、source RX1/3/4/6/8与days1/2/3，以及已验证L_s/U_s/V物理划分和标签映射。训练比例0.1，不改变数据契约或target输入包，不触发重复builder验证。训练source-screen-only入口不构造目标loader；单一源域V选模保留，周期测试使用预登记固定epoch自身快照。预测完成固定后才启动独立CPU scorer读取truth；训练只读取记录覆盖数，不消费准确率。

远端普通用户`szu2070436088`，项目`/home/szu2070436088/2510044040/CV-SincNet`；Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。唯一launch owner为当前主Agent。run ID=`a1_extended_s392005_20260910_r1`，输出`runs/<run-id>/<row>`，日志`logs/<run-id>`，CWD为本次不可覆盖release。输入包与truth路径由`BASE_RUN`和真实矩阵展开，保存在`effective_matrix.json`及各行`launch_config.json`。

启动命令：`<python> -u <release>/code/scripts/run_a1_extended_budgets.py --project-root <project> --run-id a1_extended_s392005_20260910_r1 --detach`。每GPU最多两个训练进程，训练PID内重建独立模型执行周期预测，不额外占用训练名额。已有dispatcher的16行均已启动，无待入队行；新增dispatcher只管理本次三行。

预期产物为各固定epoch的`epoch_NNN_ssdg.pth`、`target_epochs/ENNN/predictions.json`、`score.json`、`evaluation_scope.json`、最终`final_ssdg.pth`、完整训练指标和源V记录。缺少任一期或四场景覆盖不足不能标为完成。

## 技术停止与验证

仅协议越界、错误数据/CWD、输出碰撞、无法产生合法预测、错误scorer连接、系统性非有限梯度保护、无成功更新等已登记直接正确性故障阻断所属row；不因低性能停止。不干预其他run，不覆盖失败产物，不自动修改预算或重启同一root。

本地19项聚焦测试PASS，覆盖拉伸时钟、阶段边界、scratch约束、旧120/160预算回归、20轮执行和终态闭合、独立预测评分与RNG恢复。独立P0/P1只读审查PASS。真实入口检查与远端启动证据在后续状态段追加。

本地真实训练入口PASS：X2_E400、X2_E600、R3_CLEAN_RX_E400各4次有效optimizer更新，无checkpoint加载；E201及中后期检查参数更新与有限梯度PASS，证据为analysis/a1_extended_local_execution.json。

## 发布与启动

### 2026-09-10待启动R3改派

用户要求将R3安排在最早可用GPU。12:17只读核实GPU7因旧E2退出已有一个空位，R3仍唯一排队，X2两行仍为原PID。改派GPU0→GPU7，其余方法、预算、数据、seed、测试频率及输出root不变。

原dispatcher不支持动态修改pending GPU；使用本地版本化的reassign_a1_pending_gpu.py --gpu 7接管调度。仅暂停并退出核对所有权后的原dispatcher839257，新调度器ready后再交接；保留训练PID839936、839941，绝不重新启动X2。新调度器按同一原release构造R3命令，独占创建输出目录，GPU7容量不足则继续等位。继承进程无法取得exit code时明确记UNKNOWN，通过原complete_row检查最终checkpoint与全部固定期次判断产物完整性。

3项聚焦模拟测试PASS，覆盖仅修改pending GPU、成功交接不向训练PID发信号、replacement未ready时恢复原dispatcher。初次Windows测试缺少Linux信号常量，已在测试fixture显式模拟；运行脚本限定N607 Linux。独立P0/P1审查PASS。正式切换后核对新dispatcher、两条原X2 PID、R3唯一PID和GPU7绑定。

VERIFIED：训练提交1b3c0002eeb2c6bd9c283d45d44bcb99c865eccd已push并独立核对GitHub分支OID。release=/home/szu2070436088/2510044040/CV-SincNet/releases/a1_extended_1b3c0002，归档SHA256=09cf571d06c15ea83cd3e17042737c179c9da121f36bffb412c7c0fe6b408018，传输校验与远端编译PASS。发布前空闲磁盘约7.77TB。

远端三行真实入口与后期梯度检查PASS，每行4次有效optimizer更新。dispatcher PID839257；X2_E400 PID839936/GPU6、X2_E600 PID839941/GPU3均已核对PPID、CWD、argv、run-root和scratch初始化日志。R3_CLEAN_RX_E400已在同一dispatcher等待GPU0空位；不停止任何健康实验。当前GPU计算进程16个、每卡2个。

实际argv确认E400从E200、E600从E300开始，每20轮测试一次。训练L_s/U_s/V分别6300/56700/27000，label/pseudo分别260/140、390/210。目前为RUNNING，不是训练完成或性能结果；来源结论仍需完成后由自身最终checkpoint元数据闭合。

启动后独立复核VERIFIED：两行均已完成E1，日志由约6.1KB增长至11.8KB；X2_E400首轮130.2秒、X2_E600首轮126.4秒。PID与GPU占位仍正常，尚无本次目标测试结果。证据为analysis/a1_extended_remote_progress.json。
