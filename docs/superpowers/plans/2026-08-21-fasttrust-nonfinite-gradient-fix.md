# FastTrust非有限梯度修复与重发布计划

**目标：**定位并修复FastTrust/MUSE候选在CUDA训练约E17后全部跳过优化器更新的问题，完成无query真实GPU验证后，以新run ID重发原同seed、每GPU两个实验的16行矩阵。

**约束：**保持Phase1四角色比例`0.07/0.63/0.15/0.15`、seed`392002`、E200、CORE90同款LEO_WEAK增强、final-only checkpoint及训练后clean+三种LEO弱信道测试。旧run及其产物只读保留。

## 任务1：关闭无效run并固定故障证据

- 精确核对调度PID、后代PID、CWD、cmdline与run root绑定。
- 仅终止该run进程树，独立回读进程消失、GPU释放、旧产物保留。
- 将旧run标记为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

## 任务2：追踪首个非有限梯度来源

- 审查GradScaler、unscale、梯度有限性检查与MUSE共同损失路径。
- 构造最小可复现输入，逐项检查loss、参数梯度和异常算子。
- 记录控制组与MUSE共同路径差异，避免根据日志表象猜测。

## 任务3：TDD实施最小修复

- 先添加可稳定复现非有限反向梯度的失败测试并确认RED。
- 仅修改直接根因，保留FastTrust路由、协议和矩阵定义。
- 运行聚焦测试、相邻回归、语法检查和launcher检查并确认GREEN。

## 任务4：真实N607 CUDA无query验证

- 同步已提交代码到新的不可覆盖release目录。
- 使用真实checkpoint和Phase1 source角色运行多批次CUDA smoke。
- 必须同时满足：loss有限、梯度有限、`optimizer_step_applied=1`、参数发生非零变化、`query/truth reads=0`。

## 任务5：重发布实验矩阵

- 使用新run ID和新output root，保持原16行同seed消融矩阵。
- 每GPU两个训练实验；GPU3承载U128+U384，其余GPU各U256+U256以均衡步预算。
- 启动后核对PID/CWD/cmdline/GPU/log增长及首轮实际更新率。
- E200训练后由launcher自动执行clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`测试。

## 任务6：记录、提交和远端读回

- 更新旧run终止报告和新run预登记/启动报告。
- 只stage本次代码、测试、计划、配置/launcher和报告。
- 提交、自动push并独立确认远端分支OID等于本地HEAD。
