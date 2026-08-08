# Phase1 GI-EpiOR冻结设计

状态：`DESIGN_FROZEN / IMPLEMENTING`；目标模式：`GOAL_MODE=ACTIVE`。

1.候选名称：`GI-EpiOR`（Gradient-Isolated Episodic Open-set Rejector）。
2.输入只来自每fold冻结GeoSat-C的`z_id`与`tx_logits`；Sinc、主干、投影和分类头全部不更新。
3.每个source TX按canonical physical ID确定性50/50拆为reference/query，两集合物理ID严格互斥。
4.inner episode每次整TX排除；该TX不进入prototype、MAD、cache或known query，只作为`episodic nonregistered negative`。
5.定义`d_c=(1-cos(z,mu_c))/(MAD_c+eps)`，排序后`g=[d1,d2-d1,d1/(d2+eps)]`。
6.正式head固定为`3->8->1`MLP，输入stop-gradient；只用balanced full-batch BCE更新head。
7.训练常量：seed`7281105`、Adam`lr=1e-2`、`weight_decay=1e-3`、200epoch。
8.唯一拒识边界：`sigmoid(MLP(g))>=0.5`；禁止quantile、proxy/held校准和阈值扫描。
9.`p_local`恒为冻结C logits argmax；registered reject/defer在known accuracy中按错误计数。
10.无训练消融只报告`d1/(d2+eps)`连续分数，不设阈值、不构成第二候选。
11.外层矩阵为6个leave-one-known-TX-out fold；每fold一个冻结C输入和一个GI-EpiOR head。
12.ManyTx proxy20与外层held-TX全部零fit、零校准、零选择；它们只能由独立评分阶段读取。
13.门G1：角色/整TX/物理ID互斥、finite、identity梯度0、head梯度非零。
14.门G2：reject计错后的clean overall、min-class、min-RX、min-day相对C均不低于-2pp。
15.门G3：三种LEO的mean/floor/strict mean/strict floor相对C均不低于-2pp。
16.门G4：外层held 6/6达到FAR不高于5%、safe rejection不低于95%；proxy不能补偿。
17.门G5：bundle含`p_C/e_epi/d_class/mu/rho`，eager与TorchScript差不高于`1e-5`并回执成本。
18.任一门失败即`REJECT`；不调权、不扫阈值、不选有利fold、不追加对齐。
19.本轮仅为Phase1 source-only开发证据，不构成真实unknown、K-shot、Phase3协同或在轨声明。
20.独立裁决：`CHOICE=B; P0=0; P1=2; ALLOW_IMPLEMENTATION=YES; ALLOW_RELEASE=NO`，关闭两项P1后复核。
