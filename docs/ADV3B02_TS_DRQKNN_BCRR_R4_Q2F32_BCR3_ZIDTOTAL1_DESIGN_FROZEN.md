# ADV3B02-TS-DRQKNN-BCRR/r4-q2f32-bcr3-zidtotal1 DESIGN_FROZEN

## 状态

- candidate：`ADV3B02-TS-DRQKNN-BCRR/r4-q2f32-bcr3-zidtotal1`
- parent：`ADV3B02-TS-DRQKNN-BCRR/r3-q2f32-bcr2-zidtotal1`
- parent commit：`5277be456cca8e1fe261cb6ad53b68abe1da43da`
- feasibility verdict：`MERGE / P0=0 / P1=0`
- DATA_PROTOCOL=`PRESENT_REUSED / NOT_REVALIDATED`

## 唯一首源与冻结delta

r3完整125在N607的Stage2-C 26类状态构建中触发`affine BCR INT8 audit gate failed`，失败前没有prediction；该run没有性能结果。本revision只把BCR权重由固定两平面逐类对称INT8残差编码扩展为固定三平面。FP64 teacher权重、ridge/LOO、BCRR `omega`、DA、qKNN、四臂、scorer、数据和runner调度均不改变。

设`D0=0`。对`j=1,2,3`固定执行：

```text
Rj = W - D(j-1)
Qj = Decode(INT8(Rj), FP16 per-class scale)
Dj = float32(D(j-1) + Qj)
```

第三残差必须相对实际float32 `D2`计算。部署和query必须按`(Q1+Q2)+Q3`的同一float32结合顺序重建。所有K、scene、类数和注册状态固定保存三组`[160,C] int8 codes + [C] little-endian FP16 scales`；不得动态省略、增加或切换平面。

## 固定数值合同

- rounding：`numpy_rint_ties_to_even`
- clip：`[-127,127]`，禁止`-128`
- zero residual scale：最小正FP16子正规数
- 禁止FP32 weight/residual sidecar、plane4、target-derived阈值和量化fallback
- receipt绑定三平面顺序、六个array SHA、dtype、shape、canonical class order、round、clip、scale floor和实际wire bytes

## 可行证据与资源

回收首失败row的三scene support-only本机CUDA诊断中，两平面最坏logit误差为`1.03e-5`；固定三平面为`5.77e-8～6.58e-8`，三scene均为BCR top1=`1.0`、any/large flip=`0/0`。最难scene的teacher margin为`4.04e-5`，margin/error最小比为`672`。该结果只证明技术可行性，不是held性能。

BCR wire固定为`3*(160*C+2*C)=486*C B`；`C=26`时为`12,636B`，相对parent增加`4,212B`，估计最大总state不超过`163,903B<256KiB`。每query的BCR矩阵乘MAC不变；state构建每次新增`160*C`乘法和`160*C`加法，并在真实smoke/完整125记录实际延迟。

## 冻结文件范围

- `code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`
- `tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`
- `code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`
- 本设计文档及活动/新run报告

不得修改数据、scorer、authority、共享发布系统、DA/qKNN/BCRR公式或四臂定义。

## 必测与立即停止

专项测试必须覆盖第三残差的实际`D2`和加法顺序、六个array篡改、K1/K5/K10、before C6及after C11/C16/C26、置换等价、Stage2-C旧qKNN prefix、对象/字节共享、零权重、极小残差、scale floor、clip、ties-to-even、NaN/Inf、真实checkpoint无query smoke、`12,636B` wire和总state门。

任一正式support state出现BCR top1<`1.0`、any flip>0、large flip>0，或需要plane4/动态plane/阈值/fallback/FP32 sidecar，或总state超过256KiB，立即停止本revision。新完整125若再次出现系统性零prediction故障，runner必须健康止损，禁止自然失败。
