# P1-GD-ProtoNLL冻结设计卡

状态：`LOCAL_VERIFIED_V3_PENDING_IMPLEMENTATION_REVIEW`。上一版独立复审结论为`P0=0、P1=0、ALLOW`；F1G E009暴露精确零范数feature方向未定义，当前已按后续独立科学复核`P0=0、P1=0、MERGE`完成最小v3修订。本卡只定义Phase1 source-only C/G训练与后冻结连续几何诊断，不构成开放世界拒识、未知能力或晋级结论。

## 冻结机制

- 起点与共同项：每折C/G都从同一GeoSat-C`training_final_only`检查点严格加载，使用同一F1C本地4类L、同一40epoch、同一优化器新状态、同一`lambda_sat_cons=.10`和同一既有场景序列`(epoch+batch_idx-2)%3`。不得为G设置独立场景计数器；C完全不启用新增GD项。
- G唯一新增项：对单次卫星LEO forward的`feat_joint`先检查有限性，再仅过滤精确`||z_i||₂=0`的方向未定义行；不得使用eps、近零阈值、top-k或开关。既有`id_backbone.cls_head.head.weight`每行必须有限且范数严格大于0。对valid feature与head逐行L2归一，`a_ic=16*<z_i/||z_i||,w_c/||w_c||>`；`ell_c=mean_{i:y_i=c,valid_i}(1-softmax(a_i)_c)^1[-log softmax(a_i)_c]`。
- 每个G辅助批过滤前必须有本地4类，过滤后每类至少1条valid feature，否则在backward、receipt、q和EMA更新前fail-closed。对当前场景`s`使用旧权重：`L_GD=3*sum_{c=0}^3q_t[c,s]ell_c`，总损失仅为既有共同base损失加`.10*L_GD`。共同base和sat-cons仍使用完整batch；不得读取clean特征/logit、teacher、RX/domain标签、U、V、proxy或held。
- 初值`q0=1/12,barl0=0`。总损失反传后，停止梯度地对四个活跃格更新`barl[c,s]=.95barl[c,s]+.05ell_c`，再令12格`q=softmax(barl)`；不在同批重算辅助损失。
- 首个有效G批仅做一次未缩放`base`对`.10L_GD`的shared encoder与精确head梯度范数/余弦审计；其符号只诊断，不改权重、采样、优化器或选择。

## 后冻结连续几何

- 仅以封存检查点后的L真标签拟合：float64逐样本L2，`n_c>1`，`s²_cj=sum_i(z_ij-mu_cj)²/(n_c-1)`，`s²_pool,j=1/4 sum_c s²_cj`，`v_cj=max(1e-6,.9s²_cj+.1s²_pool,j)`。
- 对任何仅评分特征输出连续`u=log(4)-logsumexp_c(-NLL_c)`；没有阈值、拟合或选择，且formal训练路径不读V/proxy。

## 矩阵、终态与淘汰

固定12臂、每GPU不超过2：GPU0:F1C/F5G；1:F1G/F5C；2:F2C/F6G；3:F2G/F6C；4:F3C；5:F3G；6:F4C；7:F4G。每个G终态必须封存本地4TX×3场景的12格`total_rows/valid_rows/zero_rows/valid_loss/finite/analytic_nonzero_logit_gradient_witness`；批级和格级计数必须闭合，每个批的四类valid覆盖必须成立，zero只作诊断且不要求为0。首批审计和每批EMA更新仍为硬门。后冻结仅按已冻结的clean六折四floor、18个LEO格四floor、18格等权overall、每折三场景等权overall和逐折proxy连续诊断执行非补偿门；任何失败永久淘汰，不以proxy补偿。

## 追溯记录

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|GD-01|冻结卡：C/G|C/G逐批同一既有LEO序列，G仅`.10`GD焦点项|`train_ssdg.py`,`phase1_gd_proto_nll.py`|verified|跨epoch序列测试与v2 focused pytest|无新head/阈值/对齐|
|GD-02|冻结卡：DRO|旧q、后反传EMA、全12softmax、4类必现|`phase1_gd_proto_nll.py`,`train_ssdg.py`|verified|focused pytest|单场景固定乘3，避免active-mean抵消|
|GD-03|冻结卡：数据边界|aux只读L单次LEO，U/V/proxy不入优化|`train_ssdg.py`,`test_phase1_gd_proto_nll.py`|verified|静态检查与lite_d无query烟雾|V仅现有final-only评估|
|GD-04|冻结卡：绑定|`feat_joint`、local4、严格`training_final_only`checkpoint、精确head行与有限正L2范数|`phase1_gd_proto_nll.py`,`train_ssdg.py`|verified|v3 focused pytest|类序漂移fail-closed|
|GD-05|冻结卡：梯度|首批raw-unscaled encoder/head范数/余弦，仅诊断|`phase1_gd_proto_nll.py`,`train_ssdg.py`|verified|lite_d无query反向烟雾|None/非有限/零范数fail-closed|
|GD-06|冻结卡：终态|12格覆盖、状态更新、failure receipt|`phase1_gd_proto_nll.py`,`train_ssdg.py`|verified|focused pytest|failure写盘best-effort且不遮蔽原异常|
|GD-07|冻结卡：NLL|L-only float64对角Gaussian连续评分|`phase1_gd_proto_nll.py`,`test_phase1_gd_proto_nll.py`|verified|focused pytest|无阈值、训练未调用|
|GD-08|冻结卡：资源|固定40E final-only12臂、GPU映射与不可覆盖根|`launch_phase1_gd_proto_nll12_20260809.sh`|verified|`bash -n`与12臂dry-run|不含postfreeze|
|GD-09|v3修订：零feature|仅过滤精确零范数feature；nonfinite feature与zero/nonfinite head fatal；各类valid>=1|`phase1_gd_proto_nll.py`,`test_phase1_gd_proto_nll.py`|verified|focused pytest|无eps/阈值/top-k/开关；合法零行不进入辅助损失且梯度为零|
|GD-10|v3修订：计数闭合|批/12格封存total/valid/zero及解析梯度见证，terminal闭合且zero只诊断|`phase1_gd_proto_nll.py`,`train_ssdg.py`,`test_phase1_gd_proto_nll.py`|verified|focused pytest|缺valid类在任何状态更新前fatal；批级和终态计数漂移负测|
|GD-11|v3修订：不可覆盖运行|launcher默认run ID升级为`phase1_gd_proto_nll12_20260809_v3`|`launch_phase1_gd_proto_nll12_20260809.sh`|verified|`bash -n`与dry-run12|矩阵与参数不变|

### Postfreeze追溯

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|GD-PF-01|后冻结连续几何|GD专用clean导出按checkpoint同参数重建local4 L/U/V并逐字段闭合split/partition receipt；只forward L、V与proxy，U forward/persist均为0；Gaussian仅拟合L|`export_phase1_gd_proto_nll_features.py`,`eval_phase1_gd_proto_nll_pair.py`|verified|`py_compile`与focused pytest|逐样本精确L2；4类`n_c>1`；ddof=1；0.9/0.1 shrink；1e-6 floor|
|GD-PF-02|后冻结连续几何|以稳定logsumexp输出连续`u`，known仅为local4`source_validation_known`（V），unknown仅为`proxy_unknown`；逐折两项G-C均严格大于0|`eval_phase1_gd_proto_nll_pair.py`,`test_phase1_gd_proto_nll_postfreeze.py`|verified|连续公式、零范数与无严格改善负测|无阈值、校准、选择或补偿；外层held TX不代替V|
|GD-PF-03|后冻结绑定|四NPZ、两proxy JSON、head、manifest、checkpoint SHA、固定ManySig SHA、v3 root、arm、pair、source TX、L/V index与matrix严格绑定|`export_phase1_gd_proto_nll_features.py`,`eval_phase1_gd_proto_nll_pair.py`,`test_phase1_gd_proto_nll_postfreeze.py`|verified|U/index/head/arm/root/错SHA负测|checkpoint原receipt的dataset SHA为空时显式封存caveat；实际输入仍必须匹配冻结SHA|
|GD-PF-04|淘汰门|clean V 6/6四floor、LEO 18格四floor、每折三场景overall与18格overall非负；proxy两项6/6严格改善；任一失败永久淘汰|`eval_phase1_gd_proto_nll_pair.py`|verified|六折同matrix/root/training root/prior聚合测试|proxy非补偿；分类clean V与通用LEO随机source-only是两个独立诊断slice|
|GD-PF-05|一次性执行|固定v3训练root、12 GD-clean+12 LEO+12 proxy+6 pair共42步，LEO single/satellite/TTA none|`launch_phase1_gd_proto_nll_postfreeze_20260810.sh`,`test_phase1_gd_proto_nll_postfreeze.py`|verified|`bash -n`、dry-run=`12+12+12+6=42`|每candidate内顺序执行；12candidate按冻结GPU映射并发；不训练、不重试、不覆盖|

v3训练本地证据：`py_compile`通过；GD、CB、CP聚焦回归为31 passed，其中包括lite_d无query前后向smoke；训练launcher的`bash -n`与dry-run12通过。postfreeze本地证据：专用focused pytest为9 passed，`bash -n`与dry-run42通过。v3实现尚待独立代码复审；postfreeze未运行N607，也没有性能结果。
