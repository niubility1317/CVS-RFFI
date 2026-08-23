# ERBT-IDR M2.8局部共形翻转风险实施计划

状态：执行中；本计划是实施记录，不增加`Exclusive Minimal Experiment Workflow`之外的gate。

## 目标

在去RF32的D92 E0主基线和M2.5 B3性能分支不变的前提下，把M2.7的整行MGD96可靠性门改为逐query、逐`B0源类→B3目标类`风险判断。候选只能输出完整B0或完整B3分数行，不融合logit，不读取query truth，不以query更新状态。

## 冻结候选

- `M28-C1-B3-MGD-PAIR-POSTERIOR`：高精度top1候选。
- `M28-C2-B3-MGD-LOCAL-CONFORMAL-RECALL`：允许满足更严格条件的top2候选，用于提高对B3有益翻转的召回率。

两者共享以下support-only状态：

1. 仅从当前target support的FFT96计算MGD96，不恢复RF32；
2. 以旧类逐类中位中心的均值估计共享目标域中心；
3. 对MGD96做严格leave-one-out类别原型与类别条件非一致度校准；
4. 从`B0源类→MGD候选类`support事件构造rank、目标类和pair三级Beta-Binomial收缩后验；
5. 记录目标类LOO稳定度、目标中心径向共形分数和B3翻转margin；
6. `K<5`、全局事件不足、非有限输入或风险条件不满足时精确回退B0。

## 实施顺序

1. 先增加单元和集成失败测试，锁定support-only、顺序不变、精确B0/B3行选择、K1回退、矩阵完整性和truth-last边界。
2. 实现`stage2_m28_local_flip_risk.py`并接入通用row executor。
3. 增加screen/full125 runner、独立truth-last scorer和汇总器。
4. 运行聚焦测试、相邻M2.5/M2.7回归和真实checkpoint无query smoke。
5. 完成一次独立P0/P1审查；若出现直接P0/P1，只对原问题修复并定点复审一次。
6. 提交、自动push并核对远端OID。
7. 在N607发布不可覆盖screen；prediction闭合后再连接truth。
8. 仅当screen门槛通过时运行完整125，否则发布可复核的否定结果。

## Screen矩阵与晋级门槛

- receiver：`3-19`、`8-8`
- seed：`7282101`
- 条件：`K5/new20`、`K10/new5`
- arm：B0、B3、C1、C2
- 配对identity：4；方法row：16；场景单元：48

晋级完整125必须同时满足：

- `ΔH(Candidate-B0)≥0.002`
- `ΔH(Candidate-B3)≥0.0002`
- `N_help>N_harm`（相对B0）
- `Δmin_old≥-0.005`
- `Δmin_new≥-0.005`

低性能不属于技术失败；screen不达门槛时停止扩展完整125并形成科学否定闭环。
