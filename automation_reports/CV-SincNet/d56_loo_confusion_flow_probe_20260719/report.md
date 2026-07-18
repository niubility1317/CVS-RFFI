# D56 LOO混淆流平衡报告

## 1.状态、目标与单一差异

- 状态：`IMPLEMENTED_AND_TESTED_PRE_RUN`；operator Codex；本轮不运行125。
- 固定development cell：receiver20-1、seed713101、K10/new5、3个`leo_*_weak`场景×5fold；复用`VALIDATED_ONCE p2_min_v1`。
- 当前最强D46为before92.22%、after81.67%、new84.67%、H82.33%、forget10.56pp、min-after53.33%、min-new73.33%，仍未达到项目门槛。
- D55证明raw LOO-CE不能直接作为logit截距。D56仅把D46的support内部held预测变成离散有向混淆流，不使用CE幅值、class ID、old/new角色、scene、receiver、outer-held或query。

## 2.预注册公式

对D46 full/block两个head的每个inner-held样本，以D46已锁定的类级权重和RMS尺度形成held分数并独立argmax。若真实support类为`y`、held预测为`p!=y`，在有向图中记录边`y→p`。对每个匿名注册类`c`：

`out_c=sum_j!=c count(c→j)`

`in_c=sum_i!=c count(i→c)`

`Delta b_c=(out_c-in_c)/(K*C)`

`W_D56=W_D46`，`b_D56=b_D46+Delta b`，最后只删除类公共截距常数。因为图中每条错误边同时贡献一个out和一个in，`sum_c Delta b_c=0`。分母固定为全部held support数`K*C`，不是可调尺度；只执行一次，不回流重算图。K1/K2精确D46 fallback。

## 3.协议与禁止项

- support label只用于合法inner-held真实类和混淆边；query rows/features/labels/role/quota/true-count/global assignment均不可达。
- clean/source访问false；不恢复clean，不生成第二LEO观测，不改变capsule/split/schema。
- 所有类别使用同一公式；类标签置换时图、修正和输出同步置换；无具体TX名单。
- 禁止alpha、temperature、clip、threshold、第二arm、场景门、旧新类门、development结果后缩放及第二seed调参。
- 最终仍是单affine int8系数＋FP16截距逐query独立argmax；dense query graph为0。混淆图只在adaptation时由support构造，不进入query路径。

## 4.成功门、停止门与可观测结果

D56必须至少保持D46总体after81.67%、new84.67%、H82.33%、min-after53.33%、min-new73.33%、joint23.33%，forget不得高于10.56pp；clear/low/rain不得出现以一侧换另一侧的场景伤害；相对D46至少改变1个final prediction；INT8/FP32 before/final/margin翻转必须为0/0/0。若任一门失败，标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不跑第二seed、formal或125。

重点观测：三类总体与场景混淆、逐类old before→after、逐类new、15个outer rows、D46同折correct/wrong变化、图的out/in/net-flow分布、修正L1/L2/max、20epoch训练、量化误差、额外inner-fit资源和全部artifact SHA。报告必须保留7候选同排性能，不能只写缺陷。

## 5.实现与执行计划

1. 在D46之外重建一次相同的support inner-held full/block head，仅收集分数，不改变B20或外层fit；D46最终权重与RMS保持锁定。
2. 为混淆流、零和、类置换、rank置换、K1/K2回退、单次应用、资源闭合和tamper fail-close添加定向测试。
3. 在`ssr-gpu`下执行`py_compile`和D46＋D56窄回归；进入Git提交后，从clean detached worktree运行同一105行development矩阵。
4. 输出和本报告完成前不启动D57；D56若失败，下一轮不得扫描流强度。

本地实现已落在`code/scripts/probe_d56_loo_confusion_flow_intercept.py`，定向测试为`tests/test_probe_d56_loo_confusion_flow_intercept.py`。`py_compile`通过，D56＋D46定向回归23/23通过；覆盖混淆边流守恒、类置换、无效held score fail-close、K1/K2、固定分母、额外32次inner LDA fit及MAC/比较计数。尚未读取本轮outer结果。

## 6.版本与远端边界

Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，分支`codex/cvs-rffi-release-20260626`；根目录不是Git仓库，完成后镜像本报告。当前尚未访问N607；任何远端同步或执行必须先完成本地实现、测试、提交和N607只读preflight。
