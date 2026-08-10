# P1-HSCF冻结设计与实现追踪卡

状态：`LOCAL_VERIFIED / P0=0 / P1=0 / ALLOW-DESIGN-FREEZE_RECOMMENDED_BY_INDEPENDENT_SUPERVISOR`。本卡仅冻结Phase1 source-L-only实现，不构成性能、unknown、Phase3或N607发布声明。

## 冻结机制

对同一物理source-L批的既有clean与单LEO raw local4 logits，令`K=4`、`B0=128`、`P=I_4-11^T/4`，`a_i^v=Pℓ_i^v`，`r_i^v=a_i^v-(1/128)Σ_j a_j^v`，其中`v∈{C,L}`。唯一G辅助项为`L_HSCF=(1/512)Σ_i||r_i^L-sg(r_i^C)||_2^2`，固定`λ_hscf=0.02`；`B!=128`、`K!=4`或任何非有限值均fail-closed，不进行active-cell、样本或场景重标定。

clean整条辅助路径完全stopgrad，LEO路径、现有`model.id_backbone.cls_head.head`及shared encoder保持live。`L_HSCF=0`当且仅当存在共同`t∈1^⊥`使全部`Pℓ_i^L=Pℓ_i^C+t`，所以任意两样本head-contrast不变；共同压缩或共同旋转只在观测差分子空间恒等时可为零。common KL零集逐样本包含于该零集，但反向可有共同contrast平移；HSCF不是KL复制，也不含ICMT tail、CAGM centroid/radius/Gram、RCRMD margin、RCAT angle、RECTE cell-tail或GD EMA/prototype项。

HSCF不读RX、day或fold；class置换与`P`可交换且平方范数不变，RX同步置换无输入，因此保持等价。仅source-known L进入辅助项；U零iterate/forward/loss/backward/optimizer，V/proxy/held/target及day/fold训练、校准和选模零反馈。

C/G共同GeoSat-C`training_final_only`warm-start、physical batch/order、seed、sampler、clear/low/rain循环、40E、新AdamW、AMP和`L_base`；C辅助N/A/0，G只加`0.02L_HSCF`。首个正辅助批对未缩放项审计：LEO shared encoder及`cls_head.head.weight`VJP必须finite/nonzero；可选head bias因双中心化预期None/0，float32舍入仅按冻结`64ε·max(1,|L_HSCF|)`记作数值零；clean辅助VJP必须None/0。精确零logit与零loss合法；终态每个scene必须有正项见证、逐scene分母512和C/G共同物理/order计数闭合。

资源只增加临时`128×4`中心化logit张量及`O(BK)`归约，不增加view、模型、持久state/cache、epoch或GPU并发。后冻结仍为12臂、42步=`12 clean+12 LEO/binding+12 proxy+6 pair`，clean6/6、LEO18/18、fold/global overall和fixed400 proxy双门继续非补偿；任何后冻结结果不得回选本卡参数。

## 实现追踪

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|HSCF-01|冻结机制|固定`K=4`、`B0=128`、`D=512`和`λ=.02`公式|`code/cvsrffi/phase1_hscf.py`|已验证|128×4合成loss/VJP|不重标定|
|HSCF-02|冻结机制|clean stopgrad、LEO live、zero/nonfinite fail-closed|`code/cvsrffi/phase1_hscf.py`|已验证|合成raw-logit VJP|零logit合法|
|HSCF-03|C/G合同|解析配置、strict local4/head/batch绑定、C/G收据|`code/cvsrffi/phase1_hscf.py`、`code/SSDG/train_ssdg.py`|已验证|CLI/源码锚点、`pytest`|不修改tests或launcher|
|HSCF-04|VJP合同|首个正批encoder/head-weight非零、bias预期零|`code/cvsrffi/phase1_hscf.py`、`code/SSDG/train_ssdg.py`|已验证|未缩放128×4合成VJP|不触碰AMP/optimizer/RNG|
|HSCF-05|权限与资源|L-only接线、无新forward/state/epoch、scene终态闭合|`code/SSDG/train_ssdg.py`|已验证|`py_compile`、`--help`、`pytest 33 passed`|42步合同不改|
