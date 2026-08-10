# P1-RECTE冻结设计卡（2026-08-10）

1. 候选：P1-RECTE（Receiver-Equivariant Cell-Tail Equalization）；仅Phase1 source-only DG训练收据，不产生target、proxy或未知类性能主张。
2. 唯一可识别假设：同一source-known-train L物理样本的clean→LEO真实类margin下降在RX×local-class格之间不应长期存在相对低端尾部；只校正相对低端，不规定绝对正下降。
3. 固定格：a=(r,c)，r∈{0,…,6}、c∈{0,…,3}，共N=28；I_a={i:rx_i=r,y_i=c}，n_a=|I_a|，A_a=1[n_a>0]。
4. m_i^C=sg(l_i^C[y_i]-logsumexp_{k≠y_i}l_i^C[k])；clean原始logits整体stopgrad。
5. 令H为当前exact model.id_backbone.cls_head.head；以torch.func.functional_call对同一z_L执行H，传入全部当前参数与buffer的detach映射（buffer clone），保留H训练语义且不写live cache/state；\tilde l_i^L与已有live LEO logits逐元素数值相等，否则fail-closed。
6. m_i^L=\tilde l_i^L[y_i]-logsumexp_{k≠y_i}\tilde l_i^L[k]；δ_a=mean_{i∈I_a}(m_i^L)-mean_{i∈I_a}(m_i^C)，空格不定义δ且所有含空格pair零贡献。
7. 对固定字典序无序pair a<b，q_ab=[sg(δ_b)-δ_a]_+²+[sg(δ_a)-δ_b]_+²；仅较低δ端收到aux梯度，RX/class置换只重排同构格与pair。
8. L_RECTE=(1/378)Σ_{a<b}A_aA_bq_ab，378=28×27/2固定；不按occupied或positive pair重归一，也不使用756。
9. L_G=L_base+0.02L_RECTE；C保持同一live L_base且RECTE aux为N/A/0，common L_base的live-head输入路径逐批绑定。
10. 完整占用时零集合为全部δ_a相等，允许共同clean→LEO偏移；KL=0⇒RECTE=0，反向不成立，故非common KL、RCAT或RCRMD的拼接/换坐标复刻。
11. 与RCRMD边界：RCRMD惩罚每样本absolute positive clean→LEO margin drop；RECTE仅比较已聚合格δ的相对低端尾部，且不假定proxy几何可识别改善。
12. 权限：只读source-known-train L的TX、物理绑定rx_i、同物理clean/单LEO输出和H；不读day、fold、domain、target、proxy、held或任何U/V回流。
13. 共同trainer可构建U loader但RECTE对U零iterate/零forward/零loss/零backward/零optimizer；V仅共同只读诊断，RECTE对其零loss/零backward/零optimizer/零calibration/零model-selection反馈。
14. 资源：沿用GeoSat-C training_final_only warm-start、40E、新AdamW、AMP、同physical batch/seed/sampler与clear/low/rain scene循环；不增forward、模型、状态、cache、重采样或新视图。
15. functional H读出为每G批一次O(B×4×d)head-only计算，诚实计入资源；它不是额外model/clean/LEO forward。
16. 收据：每批记录canonical cell order、n_rc、occupancy、occupied_unordered_pair_count、positive_tail_pair_count、固定378、functional-equality与无active重归一。
17. zero特征行保留；functional/raw logits、margin、mean、δ、pair、loss或VJP出现nonfinite即fail-closed。
18. 首个positive-tail pair对未缩放L_RECTE做诊断VJP：z_L与shared encoder有限非零，exact head aux VJP必须None/0；不触碰AMP、optimizer或RNG。
19. 终态G要求每个clear/low/rain scene至少一个positive-tail pair及上述VJP；C/G的n、coverage、physical/RX/class/scene/order共同字段闭合，42步及既有clean/LEO/fold/global/proxy非补偿门不变。
