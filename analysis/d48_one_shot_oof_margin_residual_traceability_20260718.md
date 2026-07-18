# D48一次性OOF-head margin残差追溯表

状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D48-01|复用D42–D47同一fixed received-IQ development cell、old-only B20、D45 global LOO full/block融合和15个outer held折，不打开query|probe/receipt|verified|retry1 105/105行、receiver20-1/seed713101/K10-new5/3场景×5折、query0|
|D48-02|inner-held logit严格按`canonical component→各自inner-train RMS→D45 global weight`进入同一score单位|fit/tests|verified|持久化full/block/fused `K×C×C` logits并逐元素重算|
|D48-03|每类同式计算`margin=true-max_other`、`m_c=mean_r(margin)`和零和`beta_c=center(mean_c(m)-m_c)`|fit/tests|verified|手算、类置换、rank置换与公共仿射不变性测试通过|
|D48-04|beta只一次加入完整support D45 fused intercept；不得回流RMS、global weight、margin或迭代；coefficient逐bit不变|fit/tests|verified|base/final coefficient SHA相同、intercept delta与beta闭合；exact audit锁定|
|D48-05|仅称`support-supervised one-shot OOF-head margin residual`；不得称独立、无泄漏或有泛化保证的calibration|claim/report|implemented|锁定`not_independent_calibration`声明|
|D48-06|所有类同一公式；无class ID、old/new角色、scene、receiver、handle、outer-held、query、scan、clip或threshold|fit/tests|verified|exact audit、置换测试和source closure锁定|
|D48-07|K1 beta=0且逐bit回退D45；K2仍用mean且D45 unit component/global 1:1必须闭合；C<2/非有限/partition漂移fail closed|fit/tests|verified|K1/K2完整fit＋resource＋verifier链通过|
|D48-08|before state在首次new support读取前物化且不可变；final只读old+new support；query0|lifecycle/probe|verified|30/30条fit lifecycle、old/new source token、ground entry/exit和before/final formal state闭合|
|D48-09|资源保持D45 `4K+4` LDA fit、一个query state和原state/query维数；新增component scoring、affine fusion、非零`O(KC²)`margin operation及adaptation evidence内存进入审计|resource/tests|verified|K1/2/5/8/10/20手算；K8 peak numeric evidence26376B；持久化fit-audit按实际canonical compact JSON UTF-8精确计数|
|D48-10|D42–D48全链回归通过，独立代码复核无P0/P1|tests/review|verified|首轮2项P1/3项P2已修复；formal系数/JSON补强后37/130通过；固定2e-7舍入误拒改为逐元素ULP包络后38/131通过，失败105行fit audit只读复算30/30；独立复核P0/P1/P2均0且D48 27 passed|
|D48-11|真实105行协议、source、ground、state、resource、artifact与30条fit audit全部通过|probe|verified|retry1 exit0；105/105行、30/30 D48 fit、30/30 D43 fit、7个artifact SHA、query0闭合；首次失败artifact另存|
|D48-12|继承D42全部聚合、场景、floor、forgetting、joint、混淆和量化门；D45 seen-new/H不退化；min-new不低于D46；rain恢复到D42|decision|rejected|after57.78%、new56.67%、H56.23%、min-new30%；low min-new0、rain after50%；仅量化/资源通过|
|D48-13|beta至少一个fit非全0且相对D45至少改变1个final outer prediction；support margin改善不能替代outer门|decision|verified|30/30 fit beta非零；D45→D48改变135/330，98正确→错误、13错误→正确；作用充分但为负|
|D48-14|探针不生成125、不访问N607；全门通过后也仅进入另行正式化候选|scope|implemented|当前仅本地support-held开发流程|
|D48-15|每版实验必须报告全部同row指标、三场景、逐类/floor、15折、混淆、训练表现、量化、资源、artifact、相对基线变化和缺陷解释|report|verified|报告第10–19节完整覆盖；结论为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|

当前计数：`pending=0`、`implemented=2`、`verified=12`、`deferred=0`、`rejected=1`、`blocked=0`。
