# D44 full/block RMS固定融合追溯表

状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D44-01|复用D42/D43同一fixed received-IQ、15折、old-only B20，不打开query|probe/receipt|verified|105/105行完成，query0、full-K10未执行|
|D44-02|只融合full-centered与3-block-centered两套等先验LDA；不引入diagonal|fit/tests|verified|固定双组件单测通过，无候选扩张|
|D44-03|每套score只用support全行×全类的class-centered logit RMS单标量归一化|fit/tests|verified|公共score不变性、正有限RMS及精确公式单测通过；不读label/role决定scale|
|D44-04|融合权重固定1:1，不扫描weight/threshold/rank/lr/epoch/shrinkage|fit/lock|verified|常量0.5、scan count=0；D44专属artifact verifier及篡改反例通过|
|D44-05|融合后的两套affine直接合成为一套线性state，再沿用residual-int8/FP16编译；资源按before/final各2次LDA计数|fit/state|verified|K1 fallback与普通fit均按4次closed-form fit、单融合query state审计；MAC闭包单测通过|
|D44-06|类别置换等变、无类/场景专属分支、query逐样本全registry|tests/receipt|verified|置换单测通过；30条专属audit均为无分支，query0|
|D44-07|before先于new读取，ground只读，source/held互斥|D42 lifecycle|verified|真实105行lifecycle/source/ground审计全通过|
|D44-08|相对D42原始全精度基准，聚合/三类floor/逐场景/forgetting全部不退化且final floor严格改善|decision|rejected|聚合forgetting10.00pp>8.89pp；rain after76.67%<78.33%、forgetting13.33pp>10pp|
|D44-09|0个int8/FP32 before/final argmax变化与0个margin翻转；三类混淆不劣于D42 26/10/18|decision|rejected|混淆24/8/16通过，但final argmax变化1、margin翻转1|
|D44-10|探针强制identity并禁用full-K10；只有门全过才另行实现正式候选|scope|verified|receipt与metadata均锁定identity、formal=false、full-K10=false|
|D44-11|只有完整独立确认矩阵全门达标才能完成goal|confirmation|deferred|D44 development不等于目标完成|

当前计数：`pending=0`、`implemented=0`、`verified=8`、`deferred=1`、`rejected=2`、`blocked=0`。定向D42–D44回归为`56 passed`；真实105/105行已完成并全量解析。D44状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不进入正式候选、125或N607。
