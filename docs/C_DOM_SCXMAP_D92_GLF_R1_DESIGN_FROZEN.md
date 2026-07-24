# C-DOM-SCXMAP-D92-GLF/r1设计冻结

状态：`DESIGN_FROZEN / PHASE1_HELD_FALSIFIER_REQUIRED / TARGET_NOT_RELEASED`

## 可行性摘要

1.目标是同时修复D92的旧类下尾、注册后新类可达性和K1退化，不再扫描D62/D92协方差变体。
2.唯一新增DA是逐样本`SCXMAP`：由同一固定接收IQ产生的`z_dom`预测`z_id`中的接收域位移并相减。
3.Phase1只封存INT8量化的`z_dom`中心/尺度、4维接收投影、从类内中心化Phase1残差学习的4×4ridge交叉映射、4维`z_id`基和旧类地面锚点。
4.目标侧只用old-class support相对地面锚点的残差拟合一个非负连续收缩系数；new support不参与DA系数拟合。
5.K1使用六个target-old物理样本与六个地面锚点估计同一标量，不制造LOO或数学视图。
6.query的`z_dom`只变换该query；不更新状态，不读取truth、role、batch count、quota或其他query。
7.地面锚点不进入分类bank或query score，只用于support侧DA幅度识别。
8.全局头固定D92；局部头固定Student-t qKNN；弱类修正固定为有界零和support-OOF校正。
9.四臂固定为`M0`、`M_DA`、`M_HEAD`、`M_JOINT`；D62是D92内部历史基座，不增加第五臂。
10.先做未见接收机×伪新类×场景的Phase1留出证伪；只检验SCXMAP邻域是否有净纠错，不冒充D92目标性能。
11.留出门要求每个K聚合、K×场景和K×伪新类分层均有非零argmax变化、wrong→correct严格多于correct→wrong且old/new均不出现负收益，避免聚合平均掩盖负分层。
12.未通过即关闭SCXMAP，不运行目标25；通过后才生成新的联合封存bundle与完整288维实现。
13.目标开发矩阵固定seed`713102`、5 receivers×5 slices=25 jobs，覆盖K∈{1,5,10}和new∈{5,10,20}的预注册切片。
14.25 jobs仅是单seed开发证据；不代表完整125稳定性或可晋升结论。
15.目标门固定：K10 old≥92%、min-old≥85%、new5/10/20≥92/90/86%，K5相对K10退化≤5pp，K1严格改善。
16.任何bundle缺失、D92恒等复现失败、协议负测失败或P0/P1审查未清零都禁止N607目标发布。
17.构建阶段另写不可覆盖回执，绑定packet、truth和query文件SHA；predict与score必须接收预先冻结的回执文件SHA并复核全部双向承诺。
18.本地发布前必须用真实checkpoint和真实Phase1双特征archive完成support-only烟雾；只读old support，`query_access=false`、`truth_access=false`，该烟雾不是性能证据。
19.Phase1留出score即使全部门通过，也固定`target25_release_authorized=false`；目标25必须另行预注册、复核和发布。
20.精确并列logit不得依赖显示类名排序；每类使用其封存support物理ID集合生成的opaque tie token破局，prediction和score均重算并记录并列行数，保证同步类别重命名等变。

## 冻结机制

对每个样本先计算

`h=((z_dom-mu_dom)/sigma_dom)U_R`，

`c=(hA)U_I`。

Phase1封存的`U_R,A,U_I`均为类无关、接收机无关的固定低秩映射。对target-old support，使用封存地面类锚点`g_y`计算

`beta_raw=max(0,sum_i <norm(z_i)-g_yi,c_i> /(sum_i ||c_i||²+lambda*n))`，

`beta=clip(n/(n+tau)*beta_raw,0,beta_max)`。

随后`T(z,z_dom)=norm(z-beta*c)`。`beta`在Stage2-B由old support一次拟合并冻结；Stage2-C只用它变换新增support并注册类别。`beta=0`或禁用配置必须逐字节恢复原始`z_id`输入。

## 固定比较

|臂|DA|全局D92|局部Student-t qKNN|零和弱类修正|
|---|---:|---:|---:|---:|
|M0|否|是|否|否|
|M_DA|SCXMAP|是|否|否|
|M_HEAD|否|是|是|是|
|M_JOINT|SCXMAP|是|是|是|

D62、D81、D91、D92及已完成负路线只进入同协议历史对比表，不允许把不同矩阵、不同K或不同row的边际最优值拼成一个“方法结果”。
