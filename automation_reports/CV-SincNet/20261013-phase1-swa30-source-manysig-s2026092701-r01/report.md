# 单普通 cos 头：同轨迹平均的 30 轮独立检验

用户新报告将SWA列为独立机制，原模型输出的固定源RX互补检查已经完成。此候选与正在健康训练的跨RX元反馈分别消融，不合并机制、不热改旧任务；每条新轨迹仍从头30轮，总5条33300更新。不能直接跨随机初始化平均参数。本轮使用原native身份结构和单常规cos分类公式30×cos(joint160,W6×160)、177025参数，Phase2可丢弃类别权重重新拟合。

源seed2026092701，分别heldRX1/3/4/6/8，四RX fitL5040，heldRX原V5400仅冻结后评估，U关闭。原L/U/V=7/63/30，训练池标签10%。native全部随机初始化、不加载旧checkpoint；纯L clean CE、label_smoothing0，原sourceRX/day完整遍历采样seed+101、batch128、222步/轮、AdamW WD0.0001、clip10。无U、EMA、teacher、增强、辅助头或query信息。用于smoke的旧真实合法checkpoint不初始化新模型。

为了检验尾部模型平均而不是平均几乎同一点，学习率事先固定为前20轮cos0.0002至0.00005，尾10轮21至30交替0.0001/0.00005。仅这条轨迹末21..30各epoch一次快照，等权平均全部67个optimizer身份参数张量（包含普通cos权重）；final buffer及未训练参数原样复制。实际native没有BatchNorm，不使用源运行统计再校准。保存10份仅optimizer参数的紧凑快照，完成后独立重算真实均值；这不是额外训练或查询。没有根据V选择窗口、epoch、周期或LR。

同轨迹末轮FINAL和SWA均固定，分别保存同5400物理ID的source logits，然后才sourceV评分；每个推理都是一个普通native模型，无输出集成或新分类器。对原CE30同时披露LR日程不同；SWA与同轨迹FINAL的差才隔离平均效果。预登记SWA资格：五RX均值≥原合法CE30+1pp且≥FINAL+0.5pp，分别至少4/5RX非负、最差RX分别非劣1pp。完整五row后判断，不放宽、不按目标结果选方案。

若通过，三个primaryseed2026092701/02/03分别全源从零30轮、同固定LR/window和普通cos。选定SWA与预登记同轨迹FINAL配对控制，每readout逐row固定后自动clean及六practical七视图各168000，再独立truthlast；同一训练产生两个readout，训练成本不计成六个独立模型，也不隐藏两个readout推理开销。没有目标反馈调整窗口或重跑。84%仍待完整三seed实际证明，既有benchmark已暴露，不声称新盲测。

依据[权重平均原始研究](https://arxiv.org/abs/1803.05407)及用户报告；本项目AdamW、固定窗口和两轮尾部日程属于待检验扩展，不将原论文的SGD结果当当前性能保证。普通N607账号、单owner、不可覆盖run/release/root、共享capacity lock总GPU实验进程≤3/free≥12000MB。一次P0/P1、聚焦负测及真实合法checkpoint noquerysmoke、一个实际Git代码/config提交、单archiveSHA比对、remotecompile/N607preflight、postlaunch PID/CWD/cmdline/GPU/loggrowth；仅预登记系统技术/权限/输出碰撞/不可执行/预测非法等失败停止所属run，不按分数停止、不干预其他健康任务。完整矩阵、命令、环境/CWD、路径/GPU、stoprule、产物见experiment.json。

本地实际CUDA聚焦检查PASS：同轨迹算术均值、67参数、普通cos及替换9类、逐包无状态更新、5负测、完整30学习率；唯一独立P0/P1 PASS，归档入口依赖与5登记配置一致。源负结果完整保留，不与健康元反馈合并或改旧任务。
