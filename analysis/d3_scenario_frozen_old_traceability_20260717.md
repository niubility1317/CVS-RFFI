# D3 scenario-frozen-old incremental head追踪

日期：2026-07-17

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D3-01|`项目.md`9.2、9.3、10.3.1|Stage2-B旧类适应与Stage2-C新类注册同row执行，注册后保留旧类决策面|`code/cvsrffi/stage2_diag_cosine_exploration.py`；row pipeline|verified|focused pytest；development K10 new5/10/20|机制可达且证据闭环，但D3性能未过门|
|D3-02|`项目.md`7.1、7.2、7.3|只读取密封`LEO_weak` target support/query；无clean/source/query truth/role/quota/global assignment|同上；execution receipt tests|local_pass_remote_pending|协议字段和无query fit接口测试|parent receipt的完整Phase2协议字段与query-fit资源字段已fail closed核验；场景metadata不是query类别角色|
|D3-03|`项目.md`10.3.1|三个LEO场景使用对应场景support head，query逐样本对全部注册类决策|同上|local_pass_remote_pending|scenario dispatch与全注册类预测测试|场景来自密封LEO scenario槽；若实际部署无法观测场景metadata，不得称无场景Oracle|
|D3-04|`项目.md`10.3.1|开发只使用receiver`20-1`、seed`713101`、K10选择统一candidate|development launcher/report|verified|manifest及命令审计|三行统一超参数完成；未使用K5/K1或confirmation query调参|
|D3-05|`项目.md`10.3.1|参数不超过50k、每场景20epoch、状态不超过256KB、无dense query图|resource receipt；tests|verified|实际N607资源审计|C最大17,280参数、102,688B实测NPZ、20epoch/场景、无dense query图|
|D3-06|`项目.md`10.3.1|保存完整loss trace、不可变prediction、COMMIT和独立scorer|diag runner；row pipeline|verified|artifact SHA/COMMIT测试|三行各before/after完整60条trace；truth仅在两份不可变prediction后读取|
|D3-07|预登记推进门|三个new规模C-old/floor不低于D1，且每个规模`C-old - B-old >= -0.5pp`；new下降不超过2pp，new5/new10至少提升1pp|development report/scorer|rejected|同row结果表|所有规模B/C floor、old、new、H均显著低于D1；D3淘汰，不进入确认矩阵|
|D3-08|`AGENTS.md`版本与N607规则|本地先改、`ssr-gpu`验证、Git提交、再SCP和远端验证|Git/report/sync record|verified|git diff/tests/hash/preflight|提交`34a5158`、`1e69315`；远端SHA/py_compile通过|
|D3-09|用户追加要求；`项目.md`10.3.1|同时优化每个旧类floor，不能用overall平均掩盖弱类|incremental support audit；scorer/report|rejected|逐旧类margin/intrusion测试；逐TX开发结果|support intrusion为0但query floor仅68.33%/70.00%/58.33%；support保护不能替代query floor|

最高风险项：Stage2-C继承Stage2-B state必须由同row不可变COMMIT绑定，不能接受外部未绑定prior state路径。

最终状态：实现为严格设计parity，但开发性能失败，D3路线已`rejected`。下一候选必须保留D1三场景共享Stage2-B旧头，避免按场景拆分造成B old/floor先验下降。
