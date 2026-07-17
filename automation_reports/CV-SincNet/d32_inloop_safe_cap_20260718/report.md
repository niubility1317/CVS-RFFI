# D32训练期内生安全cap轻量适应实验

## 登记

- 实验ID：`d32_inloop_safe_cap_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`V1_AUTOMATION_FIELD_REPAIR_V2_PENDING`。
- 节奏：D32是D27-D29回顾后的第3轮；本轮完成后必须在D33前执行并记录新回顾。
- 目标：修复D31训练面与部署面不一致。每个Stage2-C forward从注册旧类support计算每个新类的安全非正bias，并在完全相同的带bias分数面上训练和部署；同时继续优化新类floor与旧类遗忘。
- 比较：Z0、B3诊断、C0、D32-A/B/C；6候选×3场景×5折=90行。

## 机制

旧类Stage2-B固定为B3辅助主导拼接几何和15步compact diagonal。Stage2-C冻结旧对角阵与旧权重，仅更新new suffix：

`b_j(U)=min(0,min_i(s_old(i,y_i)-s_new(i,j)-delta))`，其中`i`只遍历注册前预测正确的旧support。

|候选|Stage2-C|步数|总步数|锁定参数|
|---|---|---:|---:|---|
|D32-A|in-loop safe cap+old/new group-balanced CE|10|25|lr 0.03、delta 1e-4、anchor 0.02|
|D32-B|A+top20%新类CVaR|10|25|lr 0.03、delta 0.10、CVaR 0.35、anchor 0.02|
|D32-C|B+有限bias恢复到-4|15|30|lr 0.025、recovery 0.15、anchor 0.03|

每步重新计算cap并回滚不安全更新；最终仅按support上的新类floor、新类总体、bias接近0、较早step选择checkpoint。K=1执行质心+cap零更新旁路。最多7个新类分块，参数峰值≤2,016；无dense query图。

## 协议

- receiver `20-1`、seed `713101`、K10、5个新类、3个LEO_weak场景；沿用已验证密封support入口。
- 每个physical support只有一个已叠加LEO_weak观测；z160/FFT96/RF32只从该固定IQ确定性提取，不增加view或信道overlay。
- query为测试集且本轮不打开；无query标签、角色Oracle、真实batch类别数、类别配额或全局分配。
- clean/source不可达；Phase1 int8组件只读不可更新。当前仍是pre-formal support-only screen，不能作正式性能声明。

## 本地验证

- 新增D32 core、共享runner的candidate lock v10/fold/full/selection/receipt/CLI闭环、launcher及测试。
- D32、D31、runner、DALI、D30 envelope、D26 compact相邻测试72/72通过；`py_compile`和`git diff --check`通过。
- 随机压力覆盖2/5/10/20新类、K=1/5、A/B/C共72个状态：旧参数与旧分数前缀位级不变，bias≤0，参数≤2,016，总步数≤30，训练/部署分数面一致。
- 本地源SHA：runner `7a041be0...cb156`；D32 core `a421f914...d6dcb`。diag不修改、不上传，只核验远端`14ec9193...1ca`。

## N607计划

- 2026-07-18 07:03 CST直接preflight通过：host `dell-DSS8440`，8×RTX 3090空闲，live inventory无训练进程；检查后本地SSH/TCP22连接为0。
- 本地Git提交：`b184411c feat(stage2): add D32 in-loop safe cap route`。
- 仅同步runner、D32 core和launcher；远端SHA分别为`7a041be0...cb156`、`a421f914...d6dcb`、`1bde1442...c2efa`。远端编译、launcher语法、唯一输出不存在和继承D31/diag SHA检查通过；连接退出后SSH/TCP22为0。
- preflight与live inventory通过后，只同步runner、D32 core和launcher；其他依赖仅校验SHA。
- 远端cwd `/home/szu2070436088/2510044040/CV-SincNet`；Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 命令：`D32_GPU=0 bash code/scripts/launch_d32_inloop_safe_cap_20260718.sh`。
- output：`runs/d32_inloop_safe_cap_20260718/output/support_screen_v1`；log：`logs/d32_inloop_safe_cap_20260718/support_screen_v1.log`。
- 实际启动：2026-07-18约07:05 CST，GPU0，PID `3739951`；启动后本地SSH/TCP22连接为0。landed不等于artifact完成或性能达标。

## 结果

`support_screen_v1`完成计算后在selection聚合阶段因缺少历史兼容字段`old_score_columns_bitwise_unchanged`退出，未形成完整artifact；根因是D32把旧前缀与DALI后旧列拆成两个更准确字段，而共享聚合器仍要求历史别名。未将该失败解释为训练结果。

已补充仅代表“raw旧分数前缀注册前后位级不变”的兼容字段，同时继续独立报告`final_old_score_columns_bitwise_unchanged`；28项定向测试通过。launcher切到唯一`support_screen_v2`，runner SHA更新为`f49f3257...a93be`，等待复跑。
