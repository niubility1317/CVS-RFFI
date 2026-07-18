# D47正部锚定可靠度收缩追溯表

状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D47-01|复用D42–D46同一fixed received-IQ development cell、old-only B20和15个outer held折，不打开query|probe/receipt|implemented|runner参数与D46锁定；待真实105行闭包|
|D47-02|仅从每类每fold的合法support inner-held CE构造`d_c,r=CE_block-CE_full`；不读outer-held/query|fit/tests|verified|partition、train补集与fold CE由D46 verifier重算；D47再独立重算全部矩|
|D47-03|D45锚点严格为`z0=C×mu`，D46类观察为`z_c=K×dbar_c`；分别持久化`z0`与`zbar=K×mu`|fit/tests|verified|`C!=K`测试证明complete pooling返回D45而非错误的`K×mu`|
|D47-04|以`u_c=K×s_c²`和`tau²=max(0,Var_c(z_c)-mean_c(u_c))`做正部矩估计；收缩因子`a_c=tau²/(tau²+u_c)`|fit/tests|verified|零异质性、零方差、正异质性端点和tamper测试通过|
|D47-05|最终`zpost_c=(1-a_c)z0+a_c z_c`、`w_full,c=sigmoid(zpost_c)`；`tau²=0`精确退回D45权重公式，`u_c=0,tau²>0`该类精确到D46权重公式|fit/tests|verified|两端点与严格正、逐类和为1闭环通过；不把公式端点误报为candidate state字节等价|
|D47-06|该规则仅称“positive-part anchored reliability shrinkage”；不得称为校准后的经验贝叶斯posterior|claim/report|implemented|锁定`eb_inspired_deterministic_shrinkage_not_calibrated_posterior`|
|D47-07|类标签置换等变；无class ID、old/new角色、receiver、handle、场景、temperature、clip、阈值或扫描|fit/tests|verified|标签置换、exact audit与source closure测试通过|
|D47-08|K1固定1:1；K2仅在full/block fold CE逐项相等时1:1，否则fail closed；极端sigmoid舍入为0/1时fail closed|fit/tests|verified|K1/K2、`±30`稳定和`±1000`拒绝测试通过|
|D47-09|资源复用D46的`4K+4` LDA inventory、可靠度评分与类级融合；不新增fit、optimizer step、query state或sidecar；新增`O(CK)`标量矩运算按非零保守MAC-equivalent上界计入总adaptation|resource/tests|verified|逐state拆为`6KC` fold矩、`16C+8`跨类收缩、`8C+8` post-logit/端点；K1/2/5/8/10/20手算常数测试通过|
|D47-10|D42–D47定向回归全部通过，独立代码复核无P0/P1|tests/review|verified|首轮2项P1/2项P2/1项P3全部修复；复审无残余P0/P1；最终D47+D46 37项、全链104项通过|
|D47-11|真实105行输出的协议、lifecycle、source、ground、state、resource与artifact闭包全部通过|probe|pending|待运行|
|D47-12|相对D42全部通用聚合/场景/floor/forgetting/混淆门通过；new与min-new不低于D46，rain旧类/forgetting至少恢复到D42|decision|pending|待真实结果|
|D47-13|D47相对D46至少改变1个final held预测；若`tau²=0`使其退回D45或全部预测不变，则机制拒绝且不加第二arm|decision|pending|待真实结果|
|D47-14|before/final int8-FP32 argmax变化与margin翻转均为0|decision|pending|待真实结果|
|D47-15|探针不生成125、不访问N607；仅在另行正式化候选且开发门全过后生成125 handles|scope|implemented|当前仅本地support-held开发流程|

当前计数：`pending=4`、`implemented=3`、`verified=8`、`deferred=0`、`rejected=0`、`blocked=0`。
