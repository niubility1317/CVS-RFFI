# P1-CAGM冻结设计与实现追踪卡

## 冻结目标

P1-CAGM（Clean-Anchored Class Geometry Matching）以GeoSat-C`training_final_only`checkpoint为共同warm-start。六个LOTO fold各有C/G两臂，共12臂；两臂保持相同的物理样本、batch顺序、seed、sampler、40E、新AdamW/AMP初态、单次LEO forward、`(epoch+batch_idx-2)%3`场景循环与共同`L_base`。C只训练共同base；G唯一额外加入固定CAGM项，不增加forward、重采样、EMA、状态、阈值、RX/day/domain输入、U/V/proxy/held训练访问或选模。

对每个L标签batch，令`z=feat_joint`。先对clean和LEO的`z`及其float32逐行范数做有限性检查；任一非有限值立即fail-closed。仅辅助项使用联合掩码`M=(||z_clean||>0)&(||z_leo||>0)`；精确零范数行只从辅助几何统计中排除，`L_base`、CE和共同KL仍覆盖整个batch。对每个类，辅助有效行数必须`n_c>=2`。对`M`行，`h=z/||z||`，clean侧完全detach；`a_c=normalize(mean(h_i))`，任一质心范数非有限或精确为零立即失败。定义

```text
r_c=mean_i(1-h_i^T a_c)
g_cd=a_c^T a_d
L_CAGM=[sum_c(r_c^leo-sg(r_c^clean))^2+
        sum_c<d(g_cd^leo-sg(g_cd^clean))^2]/10
L_G=L_base+0.02L_CAGM
```

不按valid、类、term或active重缩放。辅助梯度只能到LEO侧共享encoder；首次有效G batch封存raw、unscaled辅助VJP：encoder必须finite且非零，exact classifier head的辅助梯度必须为None或精确零，这正是预期的head无辅助梯度。

## 与既有路线的正交边界

- CCPC：不做样本级cross-view对齐或对比学习。
- PAMR：不使用margin、head weight、hinge或class weighting。
- ICMT：不读取logit、不做margin-tail收紧。
- GD：不维护DRO、EMA、prototype state、focal或重加权。
- CB：不做分类边界focal项。
- CP：不做梯度投影、冲突消解或梯度修改。

后冻结42步复用门仅作为未来独立工作，当前明确`deferred`；本轮不实现、调用或暗示其已经执行。

## 追踪矩阵

|ID|冻结要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|
|CAGM-01|固定`feat_joint`球面半径+Gram公式、`/10`和`.02`|`code/cvsrffi/phase1_cagm.py`|verified|公式手算、标签置换、旋转/非等距测试通过|clean统计完全detach|
|CAGM-02|float32范数有限性、联合零掩码、质心和`n_c>=2`fail-closed|`code/cvsrffi/phase1_cagm.py`、`code/tests/test_phase1_cagm.py`|verified|zero/nonfinite/质心zero/覆盖负测通过|零行仅排除aux|
|CAGM-03|严格GeoSat-C final-only warm-start、local4标签和数据顺序绑定、新AdamW|`code/cvsrffi/phase1_cagm.py`、`code/SSDG/train_ssdg.py`|verified|focused测试通过|模型权重唯一加载，optimizer/RNG不恢复|
|CAGM-04|同一clean+单LEO forward、C/G相同三场景和batch序列|`code/SSDG/train_ssdg.py`、`code/cvsrffi/phase1_cagm.py`|verified|sequence/源码测试通过|不改变ICMT行为|
|CAGM-05|每batch/scene计数、4 radius+6 Gram、有限性和终态闭合收据|`code/cvsrffi/phase1_cagm.py`、`code/SSDG/train_ssdg.py`|verified|receipt闭合/篡改负测通过|C字段N/A或零|
|CAGM-06|raw auxiliary encoder VJP非零且head无aux-gradient|`code/cvsrffi/phase1_cagm.py`、`code/SSDG/train_ssdg.py`|verified|lite_d no-query smoke通过|只诊断，不改梯度|
|CAGM-07|六fold×C/G、40E、公平资源的12臂launcher|`code/scripts/launch_phase1_cagm12_20260810.sh`|verified|`bash -n`、dry-run12通过|无postfreeze调用|
|CAGM-08|CAGM及GD/ICMT/CB/CP窄回归|`code/tests/test_phase1_cagm.py`|verified|串行pytest共49项通过|不构成性能结果|
|CAGM-09|后冻结42步复用门|不在本轮文件集合内|deferred|不执行|仅记录未来边界|

## 完成判定

本卡的`verified`仅表示本地实现和窄验证通过，不表示N607已落地、12臂已运行、后冻结门已执行或存在性能结论。

本地验证：`python -m py_compile code/cvsrffi/phase1_cagm.py code/SSDG/train_ssdg.py code/tests/test_phase1_cagm.py`通过；`python -m pytest -q code/tests/test_phase1_cagm.py`为9 passed；与GD、ICMT、CB、CP的窄回归为49 passed；`bash -n`和启动器dry-run为12臂；`git diff --check`通过。

## 研究动机边界

[Mahajan、Tople和Sharma的《Domain Generalization using Causal Matching》](https://proceedings.mlr.press/v139/mahajan21b.html)（ICML2021）将同一对象跨域匹配作为讨论域泛化因果条件的起点；[Li、Li、Li等的《A Simple Feature Augmentation for Domain Generalization》](https://openaccess.thecvf.com/content/ICCV2021/html/Li_A_Simple_Feature_Augmentation_for_Domain_Generalization_ICCV_2021_paper.html)（ICCV2021）讨论了训练中类条件特征协方差。这两篇工作仅为“共享变化下比较稳定类几何”的研究动机。P1-CAGM不复现其网络、损失、数据设定或实验协议，唯一可声明的机制以本卡冻结公式和本地实现为准。
