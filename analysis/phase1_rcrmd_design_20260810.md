# P1-RCRMD冻结设计卡（2026-08-10）

## 冻结卡（20行）

1. 候选名：P1-RCRMD（Receiver-Conditioned Relative Margin Drop），状态：`ALLOW-DESIGN-FREEZE`后的实现冻结候选。
2. 科学主张仅为：source-L接收机×类别等权的正clean→LEOmargin-drop二阶矩可能减少接收机条件退化；它不是可识别的下分位tail。
3. 训练权限仅为`L_s={(x,y,physical_id,rx_i)}`；`rx_i`必须由source-known-train物理ID绑定并位于冻结`R_s`允许名单。
4. 禁止读取`day_i`、domain、target、proxy、held或U；无target/proxy训练、选择或调参路径。
5. 每fold冻结F1C`R_s={0,1,2,3,4,5,6}`且要求`|R_s|=7`，冻结local4类别`c∈{0,1,2,3}`，故每scene有28个RX×class格。
6. 只复用同一物理L行的既有clean和单LEO前向、既有raw local4 logits；不新增forward、view、state、pair或sampler。
7. `m_i^v=logit^v_{i,y_i}-logsumexp_{k≠y_i}logit^v_{i,k}`，仅clean margin做stop-gradient。
8. `q_i=[sg(m_i^clean)-m_i^leo]_+^2`；q=0逐行合法，所有logit、margin、q、g和loss必须有限，否则fail-closed。
9. `I_rc={i:r_i=r,y_i=c}`；`g_rc=0`若`n_rc=0`，否则`mean_{i∈I_rc}(q_i)`。
10. `L_RCRMD=(1/(4|R_s|))Σ_{r∈R_s,c∈local4}g_rc`；空格不改变分母、λ、样本顺序或采样。
11. G总损失固定为`L_base+0.02L_RCRMD`；无阈值、ε、EMA、DRO、有效格重归一或性能驱动调参。
12. C/G使用同一baseline`training_final_only`、严格model keys、相同physical/RX/class/scene覆盖、输入、batch顺序和新建空AdamW初态。
13. C/G共同封存每batch的physical/RX/class/scene、`n_rc`及固定有效权重；C的aux/active/loss/VJP严格为N/A或0。
14. G单独封存每格q活动数、loss_sum和有限性；终态要求每个`r×c×scene`非空覆盖且全程`active_q>0`。
15. G首个`active_q>0`批次必须对shared encoder和exact classifier head做raw-unscaled VJP，二者均有限且非零；审计不得触碰AMP、optimizer或RNG。
16. 训练固定40E、三LEO场景循环、`final_only`、source-validation-only、无concat/augmentation/unlabeled/均衡采样；沿用共同exit8收据风格。
17. `min-RX`是主要分类端点；`min-day`和proxy双门是独立反退化端点，均不可替代或补偿。
18. 后冻结42步收据保持12clean+12LEO/binding+12proxy+6paired；通过仅可写`pending-main`，不可称已修复RX/day/proxy。
19. 资源合同：仅共同40E前向与既有AdamW；RCRMD不增加模型参数、持久状态、数据重验或GPU并发需求。
20. 禁止跨样本/跨RX配对、target/proxy训练、性能选择、重启补偿；任何权限、公式、覆盖或VJP异常均终止为无性能结果。

## 机制追溯与边界

P1-RCRMD将惩罚对象限定为同一source-L物理行的clean与LEO相对分类margin下降，并以冻结接收机×类别格等权聚合。因此，它关注的是“LEO相对clean的接收机条件退化”，而不是总体accuracy、分位数或可识别的worst-tail估计。

它与CB/CP的差异在于：CB使用卫星分类focal-CE，CP在其基础上调整梯度；RCRMD既不改变CE，也不投影梯度。与GD-ProtoNLL不同，RCRMD没有DRO权重、原型或跨批状态。与ICMT不同，ICMT在每个view内按类别均值处理低margin行；RCRMD只比较同物理clean→LEO的正margin-drop，并额外按source receiver×class固定等权。与CAGM不同，CAGM约束`feat_joint`类半径和Gram几何；RCRMD仅作用于既有raw local4 logits的相对margin。

`rx_i`的唯一入口是训练L批次的物理绑定字段。实现仅将source split receipt的`source_receivers`规范化并冻结为F1C`{0,…,6}`，绝不以`source_train_tx`派生或替代该集合；训练时拒绝任何不在名单中的`rx_i`。训练接线只抽取`rx_i`、`base_index`/`sig_i`，不向方法传入day、domain、target或proxy字段。公式对`R_s`与`rx_i`同步置换等价且无ID专属参数或权重；实际F1C运行仍锁定物理编号`0..6`。样本重排只改变收据顺序，不改变固定公式的集合结果。

## 实现与收据路径

- 核心公式、配置拒绝、source-RX允许名单、C/G共同coverage、G-onlyq/loss、VJP和terminal fail-closed：`code/cvsrffi/phase1_rcrmd.py`。
- 训练接线：`code/SSDG/train_ssdg.py`复用CAGMv2的`training_final_only`、40E、三场景和新AdamW路径；C与G写出各自`phase1_rcrmd_*receipt.json`。
- 每batch收据保存所有`r×c`的`n_rc`、cell/row有效权重和physical/RX/class/scene事件；G另存q活动与loss贡献。终态同时检查共同28格×三场景（84格）、总行数闭合、G活动q和首活跃VJP。优化loss保持原float32路径；终态对float32批loss与同一cell账本的不同求和顺序使用冻结误差界`32×float32 eps×max(1,|batch|,|cell|)`，明显账本漂移仍fail-closed。
- 本卡不含N607运行或性能结论。测试与42步launcher属于独立机械交付面；在完整P0/P1复核、Git版本化和唯一runner交接前不得启动实验。
