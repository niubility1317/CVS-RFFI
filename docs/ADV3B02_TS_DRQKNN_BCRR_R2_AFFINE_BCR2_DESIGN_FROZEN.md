# ADV3B02-TS-DRQKNN-BCRR/r2-affine-bcr2 DESIGN_FROZEN

日期：2026-07-23
状态：`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`
监督裁决：`MERGE / P0=0 / P1=0`

## 1.触发与边界

`r2-affine`在真实checkpoint no-query smoke中于BCR权重INT8审计失败，尚未产生query prediction或性能结果。3个seed、before/after、3个scene、K1/K5/K10共54个support state中，现有按类列对称codec失败15个，固定按类列仿射失败9个，固定按特征行仿射失败12个；三者均无large-margin flip，但top1未达99.5%，立即停止。

本revision只修复BCR权重的部署表示。以下内容逐字保持：Phase1 checkpoint、`z_id/z_dom`双分支、TX抑制域qKNN、固定rank2、`alpha`、逐向量仿射INT8 support bank、Student-t qKNN、BCR ridge拟合、两方向support-LOO、`omega`安全集、四臂、K、Stage2-B/C、fallback、完整125和性能门。

## 2.固定两级INT8残差codec

对support-only拟合的FP64 BCR权重`W`，按canonical class列执行同一固定量化函数：

```text
Q(X):
  scale_c = FP16(max(max_d |X[d,c]| / 127,
                         float16.smallest_subnormal))
  code[d,c] = INT8(clip(round_to_even(X[d,c] / FP32(scale_c)), -127, 127))
  decode(Q(X))[d,c] = FP32(code[d,c]) * FP32(scale_c)

plane1 = Q(W)
plane2 = Q(W - FP64(decode(plane1)))
W_deploy = FP32(decode(plane1)) + FP32(decode(plane2))
```

- 固定恰好两层，不允许根据state、K、receiver、scene、class、old/new角色或审计结果切换层数。
- 两层只持久化little-endian INT8 codes和FP16 scales；不得持久化FP32 `W`、残差或sidecar。
- `plane2`必须从FP64 teacher权重减去`plane1`实际FP16-scale解码值生成。
- canonical class列顺序与qKNN bank完全绑定，类别置换只允许等变置换列。
- receipt必须绑定codec版本、plane顺序、shape、dtype、round/clip、scale floor、四组原始字节SHA和部署审计。

## 3.合法输入与因果结构

codec只读取当前row合法support拟合出的BCR权重，不读取query、truth、role、class quota、clean/source或receiver/TX标签。`M0/M_OTHER`共享同一raw qKNN/BCR权重state；`M_DA/M_JOINT`共享同一dual qKNN state，raw与dual只分别拟合`omega`。BCR权重teacher、BCR公式与最终全注册类竞争均不变，因此该revision是部署技术delta，不是新的OTHER或第五臂。

## 4.可辨识性与决策几何

K1仍满足`alpha=0`、`M_DA=M0`、`M_JOINT=M_OTHER`，BCRR固定`omega=0`。K5/K10的域状态与r2完全相同。两级codec只提高`W_deploy`对同一FP64 BCR teacher的保真度，不使用support审计选择分类几何；正式held上的OTHER和JOINT收益仍须由完整125证明。

## 5.资源

参数0、optimizer step0。BCR权重wire为：

```text
2 * Z_DIM * C bytes INT8 codes + 2 * C * 2 bytes FP16 scales
```

当`Z_DIM=160,C=26`时严格为8424B，比单层增加4212B。完整实际state仍须`<=256KiB`，resource receipt必须从两层实际数组字节计数，不得只报公式。

## 6.最小证据与falsifier

实现后必须通过：

1.54个真实support state的BCR teacher top1=1、any flip=0、large-margin flip=0；
2.两层bytes roundtrip与部署logit从序列化数组重算一致；
3.`plane2`残差来源、ties-to-even、clip、FP16子正规scale、dtype、shape和little-endian负例；
4.任一plane code/scale、plane顺序、receipt或class列被篡改均fail-closed；
5.类别/支持顺序置换等变，Stage2-C旧前缀不变；
6.K1 identity、K5/K10有效rank/alpha、qKNN与BCR审计同时通过；
7.C=26权重wire=8424B，完整state不超过256KiB；
8.真实checkpoint support-only smoke中query/truth文件数和fit行数均为0。

任一项失败即停止，不提交、不进入N607。Windows跳过的POSIX root-grandchild-sentinel测试必须在N607 launch前实跑通过。

## 7.冻结文件

- `docs/ADV3B02_TS_DRQKNN_BCRR_R2_AFFINE_BCR2_DESIGN_FROZEN.md`
- `docs/STAGE2_METHOD_RESEARCH_GOAL.md`
- `code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`
- `code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`
- `tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`

不得修改共享BCR实现、模型、checkpoint、数据builder、authority、coverage、GEOFF/r8或scorer。
