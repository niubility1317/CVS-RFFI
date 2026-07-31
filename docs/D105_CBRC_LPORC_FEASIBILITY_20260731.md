# D105-CBRC+LPO-RC可行性结论
状态：`FEASIBILITY_REVIEW_PASS / DESIGN_FROZEN / IMPLEMENTATION_NOT_STARTED`
1.协议：只读封存Phase1聚合bundle和当前row合法support；query零fit、零update。
2.K1：DA只估共享4D系数并使用统一预锁`lambda0`；HEAD严格恒等。
3.任务平衡：Stage2-C old/new各0.5，组内类等权；HEAD不读取role。
4.功能性：`ReLU+normalize`提供可证伪的非等距作用，避免共同变换自动抵消。
5.安全性：连续LOO收缩替代整臂hard gate；`rho≈0`仍保留prediction。
6.分类头：只增加K≥2 physical-LOO零和偏置，不改INT8 bank或Student-t核。
7.归因：`M_DA/M_JOINT`共享同一DA state；`M_HEAD/M_JOINT`共享同一HEAD代码。
8.资源：DA为4D固定4轮IRLS；HEAD query只新增每类一次bias加法。
9.代码：新增独立DA与HEAD模块，不原地修改D103/D104。
10.验证：先分别完成unit/protocol/G0，再集成四臂和真实checkpoint no-query smoke。
11.证据：独立审查DA、HEAD、联合方案均为`P0=0/P1=0/P2=0`。
12.边界：此结论只允许实现，不允许release或性能宣称。
