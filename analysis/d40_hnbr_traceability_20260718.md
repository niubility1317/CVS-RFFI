# D40 HNBR设计追踪表

## 范围

- 权威设计：`automation_reports/CV-SincNet/d40_hnbr_20260718/report.md`。
- 依据：D37–D39三轮回顾、当前`项目.md`和active objective。
- 状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D40-01|复用固定LEO弱support；无clean/source/query/role/quota|Runner/support audit|verified|真实90行support/source/query audit闭合；query保持sealed|
|D40-02|D38 Stage2-B固定20步；Stage2-C为0步HNBR|core/trace|verified|core/Runner均要求20/20且Stage2-C=0；124项回归通过|
|D40-03|公开接缝复用D38 transform/decode/compile/append，不调用私有函数|D38 seam/D40 core|verified|公开接缝readonly测试通过；D40私有D38引用计数为0|
|D40-04|HNBR固定`T=18`、难负softmax、正投影移除，无额外系数|D40 core|verified|独立公式golden、stable softmax和近零fail-close通过|
|D40-05|Stage2-B所有old同步残差化；不得顺序更新|D40 fit|verified|old顺序置换等变及同步测试通过|
|D40-06|Stage2-C冻结old prefix；所有new相对old final＋其他new base同步残差化|D40 fit/state|verified|new同步/顺序测试、decoded int8 old negative spy、FP32替换反例及prefix逐bit测试通过|
|D40-07|target-old/new最终均为两级residual-int8；formal state无FP32 target方向|state/resource|verified|formal dtype、resident FP32=0、prefix/state审计通过|
|D40-08|K1无伪LOO；覆盖K1/5/10/20和new2/5/10/20|core/tests|verified|8组参数化规模测试通过|
|D40-09|对保持enrollment partition的标签置换等变|core/tests|verified|标签重命名与方向顺序等变测试通过|
|D40-10|每样本统一面对全部注册类，无query batch依赖|score/predict|verified|row split/order/argmax不变测试通过|
|D40-11|保存old→new、new→old、new-new及最低margin/floor|Runner/geometry audit|verified|真实150条D40及150条strong B3 pairwise完整，127/150新held由old取最高分|
|D40-12|六候选×3场景×5fold=90行且held physical身份matched|candidate lock/Runner|verified|真实90/90行完成；15键physical SHA跨6候选完全matched|
|D40-13|只有D40 int8可晋级，严格比较strong B3全部同row门|selector|verified|18类逐门反例均回退identity|
|D40-14|D40 int8/FP32共享同一FP32参考方向，仅部署精度不同|matched ablation|verified|core与outer fold精度/state/argmax闭环测试通过|
|D40-15|总step=20、参数≤2016、state≤256KB，计入正值HNBR support MAC|resource audit|verified|K/new规模和selector/full-K10零MAC、步数漂移反例通过|
|D40-16|本地验证、独立审查、Git提交、真实artifact和报告闭环|repo/report|verified|独立审查无blocker；124项测试通过；实现提交`bc6c3539`；90行artifact及五项SHA闭合|
|D40-17|只有完整独立确认矩阵全门达标才能完成goal|confirmation report|deferred|D40 development不等于目标完成|

当前计数：`pending=0`、`deferred=1`、`implemented=0`、`verified=16`、`rejected=0`、`blocked=0`。

最高风险是HNBR在fit类方向上制造更大角间隔，但不能泛化到outer-held物理样本；第二风险是对old方向的投影损害D38已有域适应。两者必须由真实before-old逐类和注册后old/new同row结果否证，不能用support角间隔替代。
