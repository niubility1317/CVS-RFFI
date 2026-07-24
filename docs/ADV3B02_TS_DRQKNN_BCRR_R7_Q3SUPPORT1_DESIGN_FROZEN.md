# ADV3B02-TS-DRQKNN-BCRR/r7-q3support1 DESIGN_FROZEN

## 状态与唯一delta

- candidate：`ADV3B02-TS-DRQKNN-BCRR/r7-q3support1`
- parent：`ADV3B02-TS-DRQKNN-BCRR/r6-matchedaudit1`
- 状态：`DESIGN_FROZEN -> IMPLEMENTING`
- feasibility verdict：`MERGE / P0=0 / P1=0 / P2=1`
- 唯一delta：将所有Stage2-B/Stage2-C z_id support的固定两平面INT8 codec扩展为固定三平面INT8 codec。DA、双分支qKNN决策公式、BCRR、四臂、prediction、scorer和完整125矩阵均不改变。

## 唯一首源与反事实证据

r6完整125唯一失败row为`receiver=20-1,seed=713102,K=5,new=20`。真实support-only诊断严格复现qKNN audit=`129/130`、any/large flip=`1/0`。冻结反事实为：

|反事实|top1|any/large flip|MAE|max error|
|---|---:|---:|---:|---:|
|Q2+teacher-scale|0.9923077|1/0|0.000133314|0.001354218|
|teacher+deployed-scale|1.0|0/0|0.000004512|0.000030518|
|Q2+deployed-scale|0.9923077|1/0|0.000134723|0.001352310|
|Q3+deployed-scale|1.0|0/0|0.000004803|0.000061035|

因此deployed class scale不是首源；Q2 support重建在两种scale下保留同一翻转，而第三support残差层消除该翻转。本revision不降低`top1>=0.995`门，不增加fallback，也不改变分类公式。

## 冻结候选codec

令单位化FP32 support teacher为`U`。按固定顺序构造：

```text
D1 = affine_base_decode(Q1, scale1, offset1)
R2 = float32(U - D1)
Q2 = symmetric_int8_fp16(R2)
D2 = float32(D1 + decode(Q2))
R3 = float32(U - D2)
Q3 = symmetric_int8_fp16(R3)
D3 = unit(float32(D2 + decode(Q3)))
```

第三残差必须相对实际float32 `D2`计算；部署、wire decode和审计必须按`(D1+Q2)+Q3`同一float32顺序重建。所有K、scene、类数和注册状态均固定保存Q3；禁止动态plane、FP32 residual sidecar、阈值放宽和量化fallback。

Stage2-C append时，旧类Q1/Q2/Q3、scale、offset、bandwidth及wire前缀逐字节冻结；新类按同一固定Q3 codec追加。matched audit继续以冻结旧bank decode加当前新类FP32 support为teacher，完整after teacher继续只闭合token、label、repair和new support绑定。

## 协议、可辨识性与决策几何

- codec仅读取合法Phase1 checkpoint输出和当前row support；query、truth、role、quota和跨query状态读取均为0。
- K1/K5/K10均可确定编码；K1不拟合DA或分类参数，新增平面不改变K-shot语义。
- Q3直接改变部署support向量，可改变邻居、margin和argmax；失败scene中已观测到1个错误翻转被消除。
- 本revision只修复技术可执行性，不得以codec smoke宣称DA或性能成功；科学裁决只来自新完整125的四臂同row结果。

## 资源与生命周期

每个support row固定增加`160B int8 codes+2B FP16 scale=162B`。最坏`C=26,K=10`增加`42120B`；总state必须由真实构建实测并保持`<256KiB`。query qKNN MAC不变；构建期每row增加一轮160维残差量化与重建。optimizer step、trainable parameter、query state update均为0。所有新增Q3数组必须进入canonical wire、receipt、SHA和篡改负例；不得新增authority、allowlist、validator或数据握手。

## 冻结改动范围与最小证据

冻结后只允许修改：

1.`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`：Q3 support codec、typed state、wire、receipt和append prefix；
2.`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`：Q3重建、字节篡改、K1/K5/K10、append prefix、exact-zero标签置换/碰撞和资源回归；
3.本文档；
4.活动报告和新run报告。

不得修改数据、runner、scorer、authority、DA、qKNN/BCRR公式、四臂或method lock。监督P2要求Q3 canonical字节进入既有`_zero_class_tie_key`；这只闭合完整部署payload的确定性绑定，不改变hard tie公式。

最小falsifier：真实失败rowQ3 audit仍低于0.995、large flip非0、任一Q3篡改未fail-closed、旧prefix任一字节变化、parent query logits在同一decoded state上漂移、query进入fit/state、总state超过256KiB或需要Q4/动态fallback。任一成立即`REJECT`；否则专项测试、协议负例、真实checkpoint无query smoke和独立P0/P1 review通过后提交并发布新完整125。

## 实现闭合

- core专项pytest通过，3项Windows POSIX语义测试按预期skip；`py_compile`与`git diff --check`通过。
- 真实checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`与真实`[60,2,256]`received-IQ support完成support-only smoke：S_B 2类、S_C 6类，Q3 plane=`3`，qKNN top1=`1.0`、large flip=`0`，旧Q1/Q2/Q3均逐字节保留，state=`49,721B`，query/truth打开数、heads called及fit query rows均为0。
- 独立终审：`MERGE / P0=0 / P1=0 / P2=1`。唯一P2为私有兼容helper的返回tuple标注数量与实际不符，不影响运行、schema、wire或完整125发布；按P2不得延迟实验的规则保留到后续非阻塞整理。

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`
