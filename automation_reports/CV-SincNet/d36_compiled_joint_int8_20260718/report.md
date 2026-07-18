# D36-CJIC可编译联合int8校准实验

## 登记

- 实验ID：`d36_compiled_joint_int8_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`DESIGN_LOCKED_PREIMPLEMENTATION`。
- 目标：同时提升Stage2-B注册前旧类目标域适应和Stage2-C注册后新旧类平衡，避免D34新类不可达与D35旧类过度侵入。
- 比较：Z0、D25-C0、B3、D33-FAST、D36-A/B/C；先执行K10 support-only、3场景×5个独立outer fold。query保持关闭。
- 完整公式与协议追踪：`analysis/d36_compiled_joint_int8_calibration_traceability_20260718.md`。

## 锁定机制

D36使用同一LEO_weak接收IQ的288维B3锁定拼接表征`N([N(z160),4N([FFT96,RF32])])`。Stage2-B用target-old support执行6step，Stage2-C用target-old+target-new support再执行6step；主臂为288维对角+rank-2算子，共1,440个瞬时可训练参数。旧/新target原型均采用单中心int8量化；B/C可把密封Phase1 int8旧类锚以不超过0.20的不确定度权重融合到旧类z160块，锚始终只读。

适配结束后把算子编译进每类权重并再次量化为int8。query路径不执行adapter，只对全部注册类各做一次288维dot。D36-C另用outer-fit support内部4折cross-fit生成6维score几何，固定5次class-balanced ridge IRLS学习逐样本新旧公共offset；无query拟合、角色Oracle、真实batch类数、类别quota、global assignment或dense query图。

## 候选

| 候选 | 旧域适应 | int8旧/新头 | 地面int8锚 | 新旧校准 |
|---|---|---|---|---|
| D36-A | 对角，6+6step | 是/是 | 否 | 无 |
| D36-B | 对角+rank-2，6+6step | 是/是 | 是，只读 | cross-fit常数offset |
| D36-C | 对角+rank-2，6+6step | 是/是 | 是，只读 | cross-fit 6维连续margin |

## K10开发否证门

- 注册前old≥86.67%且目标≥88%；注册后forgetting≤3pp。
- 注册后old/new/H严格超过B3的73.33%/73.33%/72.65%。
- 任一outer fold任一旧类退化≤10pp；重点报告14-7、20-19和旧类floor。
- 09f8≥50%，f608≥73.67%；全部新类physical LOO margin>0。
- 活动参数≤50k、epoch/step≤20、状态≤50kB；5新类query dot-MAC<3,456。

若K10支持筛选未过门，不打开query。若通过，只锁定一个candidate和统一超参数，再执行K1/K5/K20压力测试和后续正式多receiver/seed矩阵；K1/K5/K20不得重新选参。

## 预估资源

| 项目 | 5新类 | 20新类 |
|---|---:|---:|
| 注册类总数 | 11 | 26 |
| query dot-MAC | 3,168 | 7,488 |
| 相对B3 query dot-MAC | -8.33% | 不同类数，不直接比较 |
| 相对K10单qKNN | -82.00% | -82.00% |
| 瞬时可训练参数 | 1,440+6 | 1,440+6 |
| optimizer持久状态 | 0B | 0B |
| 预计完整部署状态 | <32KB | <32KB |

## 执行顺序

1. 新增独立D36 core和单元测试，不修改有未归属改动的diag文件。
2. 集成共享support-only runner、完整训练日志、outer-fold与full-K10 gate、资源/协议审计。
3. 本地`ssr-gpu`窄回归和合成20新类资源验证，Git提交。
4. N607直接preflight、live inventory、最小同步、SHA闭合后执行唯一K10支持筛选。
5. 回收105行以上联合矩阵、逐类/场景、量化误差、inner cross-fit、资源和RECEIPT；负路线立即封存，正路线才扩K。
