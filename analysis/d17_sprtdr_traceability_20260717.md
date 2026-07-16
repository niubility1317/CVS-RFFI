# D17-SPRTDR实现追溯

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|D17-01|项目.md§7.1/7.1.1|只接受`RuntimeAuthorizedFeatureArtifact`固定base feature，不接触clean/source或生成第二LEO状态|module/tests|verified|普通输入不可进入；base operator/runtime/checkpoint绑定；静态Oracle面检查PASS|计算feature不增加K|
|D17-02|D17设计|pair维度仅由两端train support选择，`rank<=8`|module/tests|verified|第三类极端变异不改变pair dims/stats；held2攻击不改变train dims SHA|不使用其他类或held统计|
|D17-03|D17设计|ν=3 Student-t对角密度；`mu=median`，`sigma²=.75*pooled_var+.25*(1.4826*MAD)^2+.01`|module/tests|verified|端点toy密度方向、换端变号与finite state测试PASS|全部闭式、0epoch|
|D17-04|D17设计|`m=(Q20_a+Q80_b)/2`、`g=(Q20_a-Q80_b)/2>0`、`h=tanh((L-m)/(g+eps))`、`t=.5`|module/tests|verified|靠近a/b的h正负、端点交换严格变号；非正gap不建pair|阈值只来自train support|
|D17-05|D17设计|Before旧类最多3条endpoint-disjoint old-old边；top2命中且margin在band内时执行零和修正|module/tests|verified|非贪心matching、端点互斥、immutable-base gate、零和与其他logit逐位不变PASS|幅度网格`{0,.005,.01}`|
|D17-06|D17设计|After每个new最多2个old rival；每query仅最高new与最高old命中预登记pair时修改最高new分数，旧分数逐位锁定|module/tests|verified|global immutable-base top2反例、单new修正、旧score逐位锁和rival上限PASS|不使用query角色/标签|
|D17-07|D17红队P0|old/new floor分别由outer-train OOF自动确定；不得硬编码类别|module/tests|verified|old/new分层bottom quartile；tie用physical-ID digest；floor pair要求对应floor correct严格改善|源码无类别ID|
|D17-08|D17红队P0|每fold门：Before-old>=Z0；After-old>=Before且>=Z0；After-new>=Z0；joint/H/floor不降，否则整折Z0|module/tests|deferred|当前support-inclusive veto可整路线撤销至Z0；outer L2O完整报告Z0/joint/H/逐类比较|尚未实现逐pair原子rollback和“保留安全pair”的确定性收益排序|
|D17-09|D17红队P0|strict K10 outer L2O；held变异不改pair/rival/dims/stats/threshold/amp/floor/decision SHA|module/tests|verified|5折K8/held2及极端held攻击测试PASS|held2只用于评分；decision SHA不含full artifact SHA|
|D17-10|K协议|K1 canonical Z0；K2-K4 fail closed；K5/K10 exact-K|module/tests|verified|K1无LOO/NaN/警告；K2-4拒绝；K5与strict K10运行PASS|不从K10内部切片冒充K5/K1|
|D17-11|部署协议|state只读、自哈希、语义门；状态`<50KiB`，0参数、0epoch、无dense query图|module/runner/tests|verified|tamper失败；13old+20new×D96×R8最坏估算低于50KiB；真实v3六份state逐份低于50KiB；MAC/log1p审计PASS|真实文件证据见D17-16|
|D17-12|推理协议|单物理query、全部注册类逐样本决策，无role/quota/query拟合|module/tests|verified|single-query/all-registered与多query拒绝测试PASS|无dense query图|
|D17-13|任务边界|先完成module，再由独立runner执行真实strict-K10 support-only评估；不得开放query|module/runner/tests/artifact|verified|D8b三场景K10真实v3完成，selected true Z0；`query/truth/scorer=false`|性能NO-GO，不进入125|
|D17-14|红队增补|两pair单独安全但联合退化时按pair原子撤销，优先保留floor严格收益pair|module/tests|deferred|当前联合失败直接整路线Z0|安全但保守；不得声明细粒度rollback完成|
|D17-15|红队增补|K5 parity OOF与跨fold维稳定证书|module/tests|deferred|当前幅度用逐sample LOO；outer K10 held隔离已验证|未实现parity 3/2、2/3记录级模型证据|
|D17-16|部署工件|最终serialized state实际文件低于50KiB并由外部哈希/COMMIT加载|module/runner/tests/artifact|verified|每scene Before/After保存`state.npz+metadata.json+COMMIT`；allowlist、bytes/SHA、禁pickle、语义、state SHA和固定probe逐位回放全部PASS；v3六份总79,343B，单份最大15,232B|真实artifact为`d17_sprtdr_strict_k10_v3_final`|
|D17-17|红队P0|K5/K10选择后无任何old pair/new rival时必须规范化为Before/After一致的canonical true Z0；veto失败也必须整路线true Z0|module/tests|verified|K5/K10可分离无边回归与强制veto失败回归PASS；`rank=0,force_zero=true`、空edge/rival和原因metadata均验证|禁止用数值等价但`force_zero=false`的正候选冒充Z0|
|D17-18|runner红队P0|generation=1推理必须一次归一化、一次全注册prototype scorer pass并复用old immutable base；不得重复计算old prototype却按一次MAC记账|module/tests|verified|调用计数严格为normalize 1次、prototype scorer 1次且列数等于registered class count；与split-class deterministic reference score及prediction逐位相等|pair密度直接消费同一normalized row；资源记录`prototype_mac=C_registered*D`及最多1 old+1 new pair上界|
|D17-19|真实v2 old-lock P0|prototype scorer必须类列独立；Before `C_old`与After `C_all`的相同旧prototype不得因BLAS归约路径变化而失去逐位锁|module/tests|verified|改为`einsum('nd,cd->nc',optimize=False)`固定逐列归约；D=96、13old→33registered、N=2 held2回归中After前13列与Before逐位相等，并与逐列dot reference逐位相等|仍仅一次normalize和一次scorer pass；算法MAC保持`C_registered*D`，不产生dense query图|

## 实现与结论边界

D17已完成support-only算法原语、严格outer L2O、真实strict-K10 runner和部署state round-trip。本轮不打开query、不运行125矩阵，也不声明性能GO。`support_inclusive_deployment_consistency_veto`只可撤销；失败时Before/After整路线规范化为同一canonical true Z0。选择后本来就无active pair/rival的K5/K10也立即规范化为true Z0。唯一性能证书是strict K10 outer L2O；最终三场景统一选择true Z0。
