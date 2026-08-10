# Phase1 RCMMC冻结设计卡（2026-08-11）

状态：`LOCAL_VERIFIED / INDEPENDENT_REVIEW_P0=0_P1=0 / NO_PERFORMANCE_RESULT`；本卡冻结实现合同与本地技术证据，不构成性能或晋级结论。

## 单一核心原语

对每个source-L同物理样本对`(x_clean,x_leo)`，令`T(z)=z/||z||_2`当`||z||_2>0`，否则`T(z)=0`；先用掩码选择非零行后再除法，因此零行保留、无`0/0`与epsilon。运行时仅从sealed source-split receipt读取其有序七个source receiver token，并映射为槽位`r∈{0,…,6}`；不硬编码外部receiver ID。local4类为`c∈{0,…,3}`，每个cell内以AMP外FP32流式累积`μ_{r,c}=mean(T(z))`与对称`Q_{r,c}=X^T X/n`。

clean统计完全`stopgrad`，单一G项为`D_{r,c}=2||μ^L_{r,c}-sg(μ^C_{r,c})||_2^2+||Q^L_{r,c}-sg(Q^C_{r,c})||_F^2`，`L_RCMMC=(1/28)Σ_{r,c}D_{r,c}`，固定`λ=.02`；空cell贡献可微零且不重归一。令增广矩`M=[[1,μ^T],[μ,Q]]`，则该`D`与`||M^L-sg(M^C)||_F^2`精确等价。

## 权限、共同路径与公平

只有`source_known_train`的L在40个epoch更新；U零iterate/zero-forward，V、proxy、held、target、day、fold、domain均零训练反馈。clean与单LEO逐字段绑定同一opaque physical ID及行序，cell的`n/occupancy`、C/G、scene共同绑定；训练RX仅来自source split receipt，不改变sampler、order或采样。C为N/A/0，G仅在共同`L_base+0.02L_RCMMC`上增加该项；B=128、local4、d=160、3个冻结scene、new AdamW、AMP、warm start、order与公共KL均与C相同。

## 数学边界与旧机制关系

`0≤D_{r,c}≤12`，故单批`0≤λL_RCMMC≤.24`；`.02`由该无性能的有界尺度预先限制最坏辅助贡献，不由F6、RX、day、proxy或六旧候选结果选择。RCMMC是RCAT的cell-moment严格放松：`RCAT=0⇒RCMMC=0`；cell内样本置换可使`RCMMC=0`而`RCAT>0`，不得表述为双向不可比。它不是共同KL的逐样本softmax项，也不是ICMT、CAGM全类Gram/centroid、RCRMD margin、RECTE tail或HSCF双中心raw-logit机制；这些项的零集不替代本cell一、二阶矩约束。

## VJP、资源与失败闭合

全程仅对首个`D>0`批次的原始未缩放辅助项做一次审计：LEO`feat_joint`与shared encoder必须finite-nonzero；clean`feat_joint`与exact head辅助VJP必须None或零。三scene分别只要求终态各有至少一个正`D`批和28/28共同coverage，不重复VJP。任何输入、统计、损失或backward前nonfinite均fatal。实现逐cell处理`X^T X`，禁止`B×d²`、`B×28×d²`物化及跨批cache；AMP外FP32显式对称`Q`。保守上界为`4[32Bd+4×28(d+d²+(d+1)²)]`字节，包含28 cell的clean/LEO`μ/Q/M`形状包络与一次backward前saved-tensor包络；实际CUDA`max_memory_allocated`差分留给focused test，不使用`gc`或`empty_cache`。

## 回执与可追溯性

`cvs.phase1.rcmmc_receipt.v1`只持久化标量、计数和SHA：有序receiver SHA、行序SHA、每scene/cell的`n/A/sumD/finite/zero`、VJP计数/范数及C/G共同绑定；不持receiver token、physical key、IQ、feature、`μ/Q/M`或矩阵。终态复验三scene共同84-cell覆盖、每scene正D批、G的VJP和所有账本；任一失败写data-free failure receipt并终止。实现追踪：RCMMC-01公式；-02source绑定；-03safe totalized；-04streamed资源；-05VJP；-06receipt/terminal；-07train CLI。
