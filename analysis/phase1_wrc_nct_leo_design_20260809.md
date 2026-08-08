# Phase1 WRC-NCT三场景LEO floor冻结设计

状态：`DESIGN_FROZEN / IMPLEMENTING`；目标模式：`GOAL_MODE=ACTIVE`。

1.前置条件是WRC-NCT clean v2六折完整通过：known四项下降均为0，proxy六折有正向但较弱信号，readout/runtime闭环。
2.本轮只检验冻结clean readout在LEO压力代理下的已知类floor，不重新fit prototype、MAD、R/C/E、`tau_r`或`tau`，不评价proxy/held，不进入Phase3。
3.场景固定为`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，信道实现固定`simplified_leo_residual`。
4.每fold只导出一次2000行source数据；DataLoader固定`shuffle=false,batch_size=32`，三个场景按batch确定性轮换。每个physical ID只有一行、一次LEO叠加和一个scenario，三个scenario的physical ID集合两两不交。
5.导出的source physical ID全集必须与同fold clean source全集完全相同。clean score只提供冻结R/C/E membership及同physical ID的paired-clean baseline，不参与阈值、场景或参数选择。
6.每scenario的E切片必须同时覆盖五个source TX和全部六个source RX，否则fail-closed。
7.每scenario在同一E physical子集上报告：LEO无拒识closed、固定WRC full、coverage、overall/min-class/min-RX/min-day；所有reject按known错误计数。
8.同时报告两类下降：固定WRC相对同LEO closed的附加下降，以及LEO fixed-WRC相对同physical ID paired-clean fixed-WRC的下降。
9.六fold共6条GPU export和6条CPU score；每张GPU最多1条，不重复backbone，不扫场景、seed、batch size、阈值或聚合。
10.通过门：18个fold×scenario原子格的overall、min-class、min-RX、min-day两类下降都不超过2个百分点；6折physical/scenario闭包、固定readout SHA、GI/WRC parity和输出闭环全部通过。
11.任一格任一floor门失败即`REJECT_LEO_FLOOR`，不进入Phase3；不得以其他场景、proxy或held补偿。
12.通过才把WRC-NCT标为Phase1五门完成，并仅作为Phase3单节点本地连续拒识证据输入；不得称为真实卫星或真实unknown结果。
