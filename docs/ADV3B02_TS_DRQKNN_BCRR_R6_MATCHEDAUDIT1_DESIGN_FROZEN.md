# ADV3B02-TS-DRQKNN-BCRR/r6-matchedaudit1 DESIGN_FROZEN

## 状态与唯一delta

- candidate：`ADV3B02-TS-DRQKNN-BCRR/r6-matchedaudit1`
- parent：`ADV3B02-TS-DRQKNN-BCRR/r5-q2f32-bcr3-zidtotal1-qzero1`
- 状态：`DESIGN_FROZEN -> IMPLEMENTING / NO_PERFORMANCE_RESULT`
- feasibility verdict：`MERGE_TECHFIX / P0=0 / P1=0 / P2=0`
- 唯一delta：Stage2-C append的qKNN量化审计改用与冻结生命周期匹配的teacher；部署bank、q2 support codec、BCRR、DA、qKNN四臂、prediction/scorer和完整125矩阵均不变。

真实失败row已经用同一checkpoint和enrollment-only support在本地复现。Stage2-B冻结旧bank后，Stage2-C的after批次会重新提取同60条旧support；批形状差异使0/60行与before特征逐字节相等，最大绝对差约`6.81e-4`。原审计用after重算旧特征作为全量FP32 teacher，却要求已经冻结的before旧bank匹配它，因此clear场景出现`159/160=0.99375`的top1一致率、1个小margin flip且large-margin flip为0；low-elev和rain通过，BCR审计全部通过。确定性Q3 support残差模拟仍保留同一flip，故拒绝增加第三support平面或放宽阈值。

## 冻结matched teacher

Stage2-C append保持当前全量after teacher的token、label、repair和new support绑定门。完成这些门后，仅为qKNN量化审计构造：

```text
matched_teacher_support = concat(
    decode(frozen Stage2-B old deployed bank),
    current Stage2-C new FP32 support in canonical append order,
)

matched_teacher_bandwidth = concat(
    frozen Stage2-B deployed old class bandwidth,
    current FP32 estimate for new classes,
)
```

旧support的teacher行必须直接来自旧bank实际解码前缀，不得来自after批次重算值；旧类bandwidth必须逐字节继承旧bank已部署hi/lo。新support仍使用当前合法after FP32特征，并继续完整审计现有两平面INT8编码；新类bandwidth仍由当前FP32新类support按父公式估计，再与已部署hi+lo重建值比较。所有类别沿同一Student-t qKNN公式竞争。

原after全量teacher不得被删除或降级：它继续闭合token集合、label映射、support repair、`new_support_zid`绑定及receipt。matched teacher只改变append构建期审计的参照面，不成为持久FP32 sidecar，也不参与query推理。

## 协议、状态与决策不变性

- 只读取合法Stage2-B冻结bank和当前Stage2-C support；query、truth、role、类别quota和跨query状态读取均为0。
- 旧bank的codes/scales/offsets/residual codes/residual scales、class bandwidth hi/lo和wire前缀必须逐字节保留。
- 新类继续使用现有固定两平面support codec；不得增加Q3、动态plane、FP32 sidecar、阈值放宽或fallback。
- `qknn_logits`、BCRR、domain branch、四臂和预测输出不得改变；parent正常query和exact-zero query行为均逐字节继承。
- persistent state bytes、query MAC/时延、optimizer step和trainable parameter增量均为0；仅增加构建期临时matched数组。

## 冻结文件与接口

只允许修改：

1.`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`：为append qKNN审计提供matched support和matched bandwidth；
2.`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`：补充批形状漂移、旧prefix、新类codec故障和parent不变性回归；
3.本文档。

不得修改runner、scorer、matrix、health、data、authority、method lock、DA、BCRR或qKNN决策公式。若需要改变任何正式输入、部署bank schema、prediction或资源门，停止并创建新revision。

## 最小验证与立即证伪

必须覆盖K1/K5/K10和before/after：

- after重算旧特征发生受控微小漂移时，matched audit仍只比较冻结旧prefix和新类codec；
- 三个真实失败row scene的matched qKNN top1一致率均不低于0.995且large-margin flip为0；
- 旧prefix任一部署字节、old class bandwidth或parent query logits不得改变；
- 对新support codes/scales/offsets/residual/bandwidth注入故障时，审计仍失败关闭；
- 全量after teacher缺token、错token、错label、new support错绑、NaN/Inf或repair receipt漂移时仍失败关闭；
- state小于256KiB，query/truth/apply打开数和fit query rows均为0。

出现任一场景matched audit低于0.995、large-margin flip非0、旧prefix字节变化、新support codec故障未被捕获、parent state/wire/logit漂移或协议负例被接纳，即停止本revision。

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`
