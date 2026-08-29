# Task Plan: BiSAGE-D92两阶段实现与历史D92 E0完整125验证

## Goal
在不改写历史BiNOVA结果的前提下，实现SAGE-D和SAGE-R，先闭合阶段A最小压力行，达预登记门槛后自动继续阶段B，并在历史D92 E0相同数据、seed、split和场景上完成125行truth-last验证与Git/N607交付。

## Current Phase
Phase 1

## Phases

### Phase 1: 历史资产与需求绑定
- [x] 确认用户批准独立BiSAGE实现
- [x] 定位历史D92 E0全部数据、5个seed、5个K/新类切片及scorer
- [x] 建立报告需求到代码/测试的追踪矩阵
- **Status:** complete

### Phase 2: 实施计划与失败测试
- [x] 保存详细实施计划
- [ ] 为可微D92等价、query隔离、SAGE-D、SAGE-R、S0/S1/S2和125矩阵编写失败测试（Target125绑定测试已完成）
- **Status:** in_progress

### Phase 3: 阶段A和阶段B实现
- [ ] 实现可微D92与正式D92数值对齐
- [ ] 实现SAGE-D及阶段A门槛
- [ ] 实现SAGE-R及旧类风险约束
- [ ] 实现自动续跑和125调度
- **Status:** pending

### Phase 4: 本地验证与独立审查
- [ ] 在ssr-gpu运行聚焦测试和真实checkpoint无query smoke
- [ ] 完成一次P0/P1正确性审查并定点修复
- [ ] 生成最小预登记报告
- **Status:** pending

### Phase 5: Git/N607发布与阶段A闭环
- [ ] 精确stage、commit、push并核对远端OID
- [ ] N607 preflight、单归档SHA、远端编译和启动检查
- [ ] 阶段A真值最后评分并执行门槛
- **Status:** pending

### Phase 6: 条件式阶段B与完整125
- [ ] 阶段A通过后自动继续阶段B
- [ ] 完成125行prediction和独立truth-last评分
- [ ] 报告四状态、分层指标、资源、结论并再次commit/push/OID核对
- **Status:** pending

## Key Questions
1. 历史D92 E0的“完整125”是否已有5组旧seed资产，还是只有3组screening seed？（已回答：`E0_FULL_ONLY`完整Target125使用`713102–713106`）
2. 哪个现有runner/scorer能直接复用历史capsule/split而不重建数据？
3. K1无法形成样本级cross-fit时，如何明确回退S0并保持125口径完整？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 新增独立`stage2_bisage_*`模块 | 保留BiNOVA历史可复现性 |
| 历史D92 E0资产身份优先于新confirmation seed | 用户明确要求同数据/seed配置可比 |
| 125定义为5 receiver×5 K/新类切片×5历史seed | 与D92 E0矩阵结构直接配对 |
| K1允许support-only选择器回退S0 | K1没有独立类内held-out统计，不能伪造cross-fit证据 |
| 完整125改绑`d92_e0_full_only_target125_20260812_v1`历史矩阵 | 该矩阵已真实完成125/375，满足用户“以前D92 E0跑过”要求 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Windows只读检索误路由到WSL并失败 | 1 | 停止该路径，后续固定Windows原生`cmd.exe`和`.exe`工具 |
| `cmd.exe`中带空格/管道正则引用被拆分 | 1 | 改用无空格模式、分次检索或直接读取目标文件 |
| conversation index多词参数被cmd拆分 | 1 | 改用单token检索并设置UTF-8输出 |
| conversation index输出触发GBK编码异常 | 1 | 后续命令设置`PYTHONIOENCODING=utf-8` |
| 计划占位符扫描中带空格模式再次被cmd拆分 | 2 | 改用无空格关键字分次扫描，未重复原命令 |

## Notes
- 不增加八项白名单外的gate；额外旧文档要求一律记录为`REJECTED_EXTRA_GATE`。
- 低性能是科学结果，不是技术停止；但用户明确预登记的阶段A晋级门槛决定是否扩展阶段B。
- 所有正式结果使用`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`。
