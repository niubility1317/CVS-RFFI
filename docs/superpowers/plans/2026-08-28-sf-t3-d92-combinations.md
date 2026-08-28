# SF-TAPFT t3.norm与D92组合实验实施计划

> **执行说明：**本计划按`p2_min_v1`和项目最小实验工作流实施。首轮是可证伪矩阵，不扩展为完整125确认。

**目标：**实现并比较D0/H6 Compact、S02长程、R3双delta三种仅持久化`t3.norm`的域适应方法与固定D92-E0-NORF32注册头的组合效果。

**统一因果口径：**每个数据row复用同一`VALIDATED_ONCE` support/query切片，先完成四状态prediction，再由独立scorer连接truth。域适应输出只包含`model.t3.norm.weight`和`model.t3.norm.bias`；训练期target head是support surrogate，不进入部署bundle。D92只读取合法old/new support，不读取query或truth。

**首轮矩阵：**3种候选×`N_new∈{2,10,20}`×3种LEO弱场景，共27个数据格；每格报告`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`。旧类固定6类、K=10，checkpoint固定`ADV3B02_CORE90_SOFT_E200`，D92固定E0且RF32关闭。

## 任务1：TDD实现三个候选接口

**独立文件：**
- `code/cvsrffi/stage2_sf_t3_d92_d0.py`
- `code/cvsrffi/stage2_sf_t3_d92_s02.py`
- `code/cvsrffi/stage2_sf_t3_d92_r3.py`
- 对应三个聚焦测试文件

1. 先写只允许`t3.norm`部署状态的失败测试。
2. 写D92锁定、query零访问、临时head丢弃的失败测试。
3. R3另写support-only交叉拟合D92风险、标签置换不变性测试。
4. 实现最小接口并运行聚焦测试。

## 任务2：统一配置与四状态runner

**文件：**
- `configs/stage2_sf_t3_d92_combo_n2n10n20_rx20_1_s713101_m392002_20260828.json`
- `code/scripts/run_stage2_sf_t3_d92_combinations.py`
- `tests/test_stage2_sf_t3_d92_combinations.py`

1. 冻结27格矩阵、不可覆盖run root和三候选训练规格。
2. 复用既有四状态D92预测路径和相同输入句柄。
3. prediction闭合后独立评分，写出逐格指标与资源证据。

## 任务3：本地验证、Git发布与N607实验

1. 在`ssr-gpu`环境运行聚焦协议负测和真实checkpoint无query smoke。
2. 完成一次P0/P1正确性审查；只阻断直接导致真实实验跑错或越权的问题。
3. 精确stage、commit、push，并核对远端OID等于本地HEAD。
4. N607短连接preflight、单release归档SHA核对和远端编译。
5. 按每GPU最多2个训练任务调度；启动后核对PID/CWD/cmdline/GPU/log增长。
6. 监控到prediction完整，独立scorer评分，更新中文报告并再次Git发布。

## 停止与晋级

- 仅协议/query泄漏、错误row/输入、输出覆盖、错误checkout、确定性执行故障或无prediction闭合可技术停止。
- 低性能不停止任务。
- 首轮选择满足旧类准确率、旧类floor和旧新H值不退化的最小候选；若三者均未改善，保留D0共同基线并报告负结论。
