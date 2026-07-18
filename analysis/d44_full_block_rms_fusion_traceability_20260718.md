# D44 full/block RMS固定融合追溯表

状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D44-01|复用D42/D43同一fixed received-IQ、15折、old-only B20，不打开query|probe/receipt|pending|真实105行待执行|
|D44-02|只融合full-centered与3-block-centered两套等先验LDA；不引入diagonal|fit/tests|verified|固定双组件单测通过，无候选扩张|
|D44-03|每套score只用support全行×全类的class-centered logit RMS单标量归一化|fit/tests|verified|公共score不变性、正有限RMS及精确公式单测通过；不读label/role决定scale|
|D44-04|融合权重固定1:1，不扫描weight/threshold/rank/lr/epoch/shrinkage|fit/lock|verified|常量0.5、scan count=0；D44专属artifact verifier及篡改反例通过|
|D44-05|融合后的两套affine直接合成为一套线性state，再沿用residual-int8/FP16编译；资源按before/final各2次LDA计数|fit/state|verified|K1 fallback与普通fit均按4次closed-form fit、单融合query state审计；MAC闭包单测通过|
|D44-06|类别置换等变、无类/场景专属分支、query逐样本全registry|tests/receipt|implemented|类别置换等变和无分支单测通过；真实receipt/query closure待执行|
|D44-07|before先于new读取，ground只读，source/held互斥|D42 lifecycle|pending|复用真实Runner审计待执行|
|D44-08|相对D42原始全精度基准，聚合/三类floor/逐场景/forgetting全部不退化且final floor严格改善|decision|pending|沿用D43的`1e-12`门|
|D44-09|0个int8/FP32 before/final argmax变化与0个margin翻转；三类混淆不劣于D42 26/10/18|decision|pending|真实15折待执行|
|D44-10|探针强制identity并禁用full-K10；只有门全过才另行实现正式候选|scope|implemented|复用复合lock/三方SHA verifier|
|D44-11|只有完整独立确认矩阵全门达标才能完成goal|confirmation|deferred|D44 development不等于目标完成|

当前计数：`pending=4`、`implemented=2`、`verified=4`、`deferred=1`、`rejected=0`、`blocked=0`。定向D42–D44回归为`56 passed`；pytest退出后的Windows临时目录清理出现既知`WinError 5`噪声，但测试进程退出码为0。
