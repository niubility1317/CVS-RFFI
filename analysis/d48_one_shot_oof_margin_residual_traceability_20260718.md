# D48一次性OOF-head margin残差追溯表

状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D48-01|复用D42–D47同一fixed received-IQ development cell、old-only B20、D45 global LOO full/block融合和15个outer held折，不打开query|probe/receipt|implemented|runner与输入锁定；待真实105行|
|D48-02|inner-held logit严格按`canonical component→各自inner-train RMS→D45 global weight`进入同一score单位|fit/tests|verified|持久化full/block/fused `K×C×C` logits并逐元素重算|
|D48-03|每类同式计算`margin=true-max_other`、`m_c=mean_r(margin)`和零和`beta_c=center(mean_c(m)-m_c)`|fit/tests|verified|手算、类置换、rank置换与公共仿射不变性测试通过|
|D48-04|beta只一次加入完整support D45 fused intercept；不得回流RMS、global weight、margin或迭代；coefficient逐bit不变|fit/tests|verified|base/final coefficient SHA相同、intercept delta与beta闭合；exact audit锁定|
|D48-05|仅称`support-supervised one-shot OOF-head margin residual`；不得称独立、无泄漏或有泛化保证的calibration|claim/report|implemented|锁定`not_independent_calibration`声明|
|D48-06|所有类同一公式；无class ID、old/new角色、scene、receiver、handle、outer-held、query、scan、clip或threshold|fit/tests|verified|exact audit、置换测试和source closure锁定|
|D48-07|K1 beta=0且逐bit回退D45；K2仍用mean且D45 unit component/global 1:1必须闭合；C<2/非有限/partition漂移fail closed|fit/tests|verified|K1/K2完整fit＋resource＋verifier链通过|
|D48-08|before state在首次new support读取前物化且不可变；final只读old+new support；query0|lifecycle/probe|pending|待真实输出闭包|
|D48-09|资源保持D45 `4K+4` LDA fit、一个query state和原state/query维数；新增component scoring、affine fusion、非零`O(KC²)`margin operation及adaptation evidence内存进入审计|resource/tests|verified|K1/2/5/8/10/20手算；K8 peak numeric evidence26376B；持久化fit-audit按实际canonical compact JSON UTF-8精确计数|
|D48-10|D42–D48全链回归通过，独立代码复核无P0/P1|tests/review|verified|首轮2项P1/3项P2已修复；formal系数绑定与真实JSON计数补强后定向37项、全链130项通过；最终复审P0=0、P1=0、P2=0，D45 HEAD/default K1/K5逐bit/JSON一致|
|D48-11|真实105行协议、source、ground、state、resource、artifact与30条fit audit全部通过|probe|pending|待运行|
|D48-12|继承D42全部聚合、场景、floor、forgetting、joint、混淆和量化门；D45 seen-new/H不退化；min-new不低于D46；rain恢复到D42|decision|pending|待真实结果|
|D48-13|beta至少一个fit非全0且相对D45至少改变1个final outer prediction；support margin改善不能替代outer门|decision|pending|待真实结果|
|D48-14|探针不生成125、不访问N607；全门通过后也仅进入另行正式化候选|scope|implemented|当前仅本地support-held开发流程|

当前计数：`pending=4`、`implemented=3`、`verified=7`、`deferred=0`、`rejected=0`、`blocked=0`。
