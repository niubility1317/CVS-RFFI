# ADV3B02-TS-DRQKNN-BCRR/r5-q2f32-bcr3-zidtotal1-qzero1 DESIGN_FROZEN

## 状态与唯一delta

- candidate：`ADV3B02-TS-DRQKNN-BCRR/r5-q2f32-bcr3-zidtotal1-qzero1`
- parent：`ADV3B02-TS-DRQKNN-BCRR/r4-q2f32-bcr3-zidtotal1`
- 状态：`DESIGN_FROZEN -> IMPLEMENTING / NO_PERFORMANCE_RESULT`
- feasibility verdict：`MERGE / P0=0 / P1=0 / P2=0`
- 唯一delta：只为finite、componentwise exact-zero的query`z_id`定义无方向qKNN解析延拓和标签无关硬平局；正常query、support、DA、BCRR、四臂、scorer和125矩阵不变。

首源已由真实run traceback固定为`after_zid -> predict_four_arms -> M0 qknn_logits -> _score_support -> _unit(query)`。`_rows`已先证明元素全finite；允许证据未读取query内容，因此不声称scene、token或truth。真实checkpoint support-only重放覆盖三场景780条after support，IQ、`z_id/z_dom`均finite，support最小`z_id`范数为`0.04403`，repair count为0，四臂state全部构建成功。

## 冻结输入边界

query API仍只接收同一query的float32`z_id/z_dom`；不接收truth、old/new role、query token、类别配额、batch类数或跨query状态。令：

```text
zero_mask[i] = all(query_zid[i,j] == float32(0.0) for every j)
```

- shape/dtype不合法或包含NaN/Inf：fail-closed；
- `zero_mask=false`且float64 L2 norm`<=1e-12`：继续fail-closed；
- 只有`zero_mask=true`的row进入本revision延拓；
- support仍使用既有`finite_exact_zero_singleton_class_medoid_v1`，不得复用query规则。

## 无方向qKNN解析延拓

对exact-zero query，冻结`cos(q,s)=0`、球面距离`d(q,s)=2`。对注册类`c`及其已部署class bandwidth`h_c`：

\[
\ell_c^{(0)}=
-\gamma d_{eff}\log h_c
-\frac{\nu+d_{eff}}{2}
\log\left(1+\frac{2}{\nu h_c^2}\right).
\]

该式是父方法Student-t qKNN在固定`distance=2`下的解析代入，不宣称为连续极限。按类`logsumexp-log(K_c)`自然消去support数量；`h_c`只能来自当前M0已封存的INT8/FP16 bank，不得读取新先验或query统计。before/after、K1/K5/K10和全部old/new注册类使用同一公式。

对每个exact-zero row强制：

```text
M0 == M_DA == M_OTHER == M_JOINT
```

四臂必须逐字节复制同一个M0解析分数。不得让`z_dom`、domain weighting或BCRR在该row产生增益，因此该row对`I_syn`贡献严格为0。混合batch中的非零row必须直接调用parent原路径，logit和prediction逐字节不变。

## 标签无关精确平局

仅当exact-zero row的最大logit出现逐位精确平局时，不使用registry位置、class handle、query token或随机数。每个并列类使用当前`Int8QKNNState`已经持有的标签无关class payload形成排序键：

1.该类对应support rows的`codes/scales/offsets/residual codes/residual scales`规范小端字节；
2.该类`class_scale_hi/lo`规范小端字节；
3.该类现有opaque support physical-token tuple按字节序排序。

按`(payload_bytes,sorted_support_token_tuple)`字典序选择最小键。payload不得包含class label、class axis或registered role。support token全局唯一，因此两个类键完全相同视为state drift并fail-closed。类轴重排后score按轴同步重排，最终物理类选择不变。非零row即使出现平局也继续使用parent预测路径，不得改变历史行为。

## 生命周期、资源与receipt

- Stage2-B/Stage2-C state构建、append和INT8生命周期完全继承parent；query不更新state。
- 新增运行统计仅为`query_zid_exact_zero_count`、`query_zid_exact_zero_rate`、`query_zid_exact_tie_count`和`zero_rows_all_arms_equal=true`；这些字段不得成为选择、回滚或性能gate。
- prediction和truth-side scorer格式不变；125矩阵、GPU调度、health策略、archive/coverage逻辑不变。
- trainable parameter、optimizer step、persistent state bytes和正常query MAC不增加；zero shortcut只减少domain/BCRR计算。

## 冻结文件与接口

只允许修改：

1.`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`：exact-zero检测、解析分数、四臂相等、平局helper和统计；
2.`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`：before/after共同使用平局helper并扩展现有runtime receipt；
3.`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`：直接专项、协议负例和parent不变性。

不得修改scorer、data、authority、method lock、support codec、BCRR fit、DA公式、矩阵或控制面。若实现需要改变核心输入、support规则、query token接口、scorer或其他arm，停止并创建新revision。

## 最小验证与立即证伪

必须覆盖：

- K1/K5/K10、before/after、normal/zero混合batch及query行重排；
- normal row四臂logit和prediction相对parent逐字节不变；
- zero row四臂有限、逐字节相同且`I_syn=0`；
- registry/class-axis重排、class label重命名和exact-tie物理类等价；
- NaN、Inf和tiny-nonzero负例；
- support exact-zero repair和tiny-support负例不变；
- 真实checkpoint三场景support bank＋合成exact-zero query的无truth prediction/scorer闭环；
- state bytes、MAC和现有INT8门不回退。

出现任一情况立即停止本revision：正常row任一字节漂移；zero row四臂不相同；精确平局随class ID或registry顺序改变；非法tiny/nonfinite被接纳；prediction不完整；scorer或现有资源/INT8门无法闭合。

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`
