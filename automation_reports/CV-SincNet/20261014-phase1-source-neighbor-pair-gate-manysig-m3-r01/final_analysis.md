# 物理邻包成对采样源诊断完整结果

状态ANALYZED；run `20261014-phase1-source-neighbor-pair-gate-manysig-m3-r01`；实际执行代码 `7bde2923fae013495c3be1038504eb59d6e44602`。完整3模型×2samplers×L/V已自然结束，技术失败/缺项0；720条计数/比率/分RX、day、source类合计独立复算VERIFIED，RANDOM的360条旧r01计数精确复现。一次归档校验、远端compile、真实checkpoint零IQ无query smoke和owned PID/CWD/argv/CPU环境/log增长均VERIFIED；日志0→269字节、原进程自然退出。

三个seed均通过预先固定的V条件：FULL覆盖≥50%、正确率≥99.5%，相对同seed RANDOM正确率下降≤0.25个百分点。未改变阈值/门控/分类头，不使用目标结果。

| 合法源角色 | 采样 | 完整门控覆盖均值 ± 模型样本 SD（%/pp） | 接收正确率均值 ± 模型样本 SD（%/pp） | 接收数 seed01/02/03 | 错误数 seed01/02/03 |
|---|---|---:|---:|---|---|
| L_s | RANDOM | 13.3598 ± 0.1629 | 100.0000 ± 0.0000 | 839/833/853 | 0/0/0 |
| L_s | PAIRS | 92.9312 ± 0.9463 | 100.0000 ± 0.0000 | 5859/5793/5912 | 0/0/0 |
| V | RANDOM | 3.4099 ± 0.0252 | 100.0000 ± 0.0000 | 923/913/926 | 0/0/0 |
| V | PAIRS | 89.9617 ± 0.7088 | 99.9959 ± 0.0071 | 24390/24069/24410 | 3/0/0 |

几何邻包在PAIRS的L/V每批及尾批均100%可见；FULL减少到约90%来自原confidence/neighbor-prediction agreement/strongagreement，机制含义是改变邻包可见性，不能称关闭temporal gate。RANDOM几何可见V3.7630%、L14.2381%，与既有结果一致。teacher weak source整体准确率在两种顺序下保持一致；默认学生/EMA参数和buffers逐项未变。相同source数据和固定shuffle分别复用，模型seed SD不是多个独立采样试验，不将33300物理包当模型seed数。

PAIR V接收错误3/0/0；RANDOM只接收923/913/926时未观察到错误，是有限样本事实，不保证绝对100%可靠。PAIR源代理正确率高且覆盖增加，但冻结的L/V模型观察不能推断真实U_s正确率、未来训练稳定性或目标DG。强noise/dropout随机种子重置相同，但重新排序后随机扰动附着的物理包改变，不宣称逐包增强完全匹配。

只有L_s/V合法标签在mask固定后用于观察；U IQ/物理ID/真值及target全部0读取。0模型、头、optimizer和buffer更新；实际CPU215.2072秒，399600包次、1572批前向。来源为固定历史OLD_GATE_COS50 33＋17/fromscratch50学生/自身实际EMA，仅观察，不作下一训练父权重。默认cos6×160/scale30、NoExplicitCFO、内部补偿保持。

完整720条原始聚合、每seed四gate及RX/day/source类指标、source条件、成本和独立复算scope见 [results.json](results.json)，固定方案与执行位置见 [report.md](report.md)。数据/权重/原始日志留N607 run，Git交付实际实现、冻结配置与正负源诊断证据。此run不是目标测试或新TX实验。

下一唯一核心假设：只改变合法U_s每轮的随机批次顺序为物理邻包随机pairs，保留旧门控/EMA/普通cos/default头，独立scratch50同路径RANDOM控制；不叠加Sinc/TIME，不依据target评分选参数。实际训练必须在独立run固定配方、第一seed源反证、预算与自动clean/truth-last之后发布；本source诊断通过本身不等于训练已启动。
