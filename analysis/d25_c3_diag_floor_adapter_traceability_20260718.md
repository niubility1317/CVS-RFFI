# D25-C3 288D拼接对角floor适配追溯表

日期：2026-07-18

状态：核心模块、独立runner接入与N607 support-only实验均已完成；核心资源合规，但性能/floor门未通过

约束来源：用户D25拼接决定、`项目.md` Phase2资源与query隔离协议、D21-M6 support-only floor/CVaR设计。

## 权重语义边界

`项目.md`中的`0.07/0.63/0.30`首先定义Phase1有标签训练、无标签训练和source validation的互斥数据比例，不是D25-C3的模态能量或默认loss权重。D21-M6曾把相同数值独立锁为`equal CE/tail CVaR/prox+forgetting`损失权重，但C3不继承这组三数作为隐式默认；调用方必须显式构造语义命名的loss配置。

## 需求—实现追溯

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|C3-01|用户决定|沿用`z160+FFT96+RF32=288D`拼接，不降维|`code/cvsrffi/stage2_multimodal_diag_floor_adapter.py`|verified|`test_transform_preserves_all_three_fixed_block_energies`|输入必须保持三块固定能量|
|C3-02|用户轻量要求|shared adapter只含288个可训练`gamma`|同上|verified|`test_stage2b_trains_exactly_288_shared_parameters_and_logs_margin`|不训练backbone或dense head|
|C3-03|D25设计|每块`gamma`零均值并clip到`[-0.35,0.35]`|同上|verified|投影与真实2-step持久化测试|消除块级能量漂移自由度|
|C3-04|D25设计|变换后每块重新归一化，固定`5/9,1/3,1/9`平方能量|同上|verified|逐块energy测试|loss权重不得充当模态能量|
|C3-05|Stage2-B|仅support full-batch更新，最多20 optimizer steps|同上|verified|step边界、full-batch trace测试|无query入口|
|C3-06|D21 floor语义|显式等权类CE、worst-25% CVaR、hard-negative margin和近端项|同上|verified|loss trace与显式配置测试|四项权重由显式`LossWeights`传入|
|C3-07|Stage2-C|默认0-step闭式append；可选最多30步只更新new suffix，总步数不超过50|同上|verified|零步/2步suffix与资源计数测试|可选suffix参数单列，不计入288 shared adapter|
|C3-08|防遗忘|Stage2-C后旧prototype prefix与shared gamma逐字节冻结，但不得把raw-score冻结声明为预测无遗忘|同上|verified|SHA、payload、old-score bitwise及argmax反例测试|实际old-support非退化/回退门由runner执行|
|C3-09|Phase2 query协议|fit/append API不含query/truth/role/quota/global assignment|同上|verified|签名与单样本评分测试|预测只接受单个feature|
|C3-10|资源审计|记录adapter参数、可选suffix参数、步数、MAC、状态与无dense query图|同上|verified|resource audit测试|正式部署上限80k/50step/256KB；epoch由runner再审计|
|C3-11|工程验证|新模块与focused tests通过`py_compile`和pytest|模块、tests|verified|29项相邻回归PASS|不提交，由主agent统一整合|
|C3-12|原子注册|Stage2-C只允许一次批量注册5/10/20类，已存在suffix时重复append必须fail closed|同上|verified|重复append回归|防止step计数重置及忽略既有new竞争|
|C3-13|epoch资源门|逐阶段及总adaptation epoch分列；≤30正式，31–45仅150%探索，>45拒绝|同上|verified|20+20探索tier与20+26拒绝测试|每个full-batch step等同一个adaptation epoch|
|C3-14|逐样本部署计算|query transform和prefix/suffix matmul固定FP32并审计临时内存|同上|verified|dtype、temporary bytes与bitwise回归|训练期gamma投影可保留FP64数值控制|

## 本地验证

- `conda activate ssr-gpu; python -m py_compile code\cvsrffi\stage2_multimodal_diag_floor_adapter.py tests\test_stage2_multimodal_diag_floor_adapter.py`：PASS。
- `conda activate ssr-gpu; python -m pytest -q tests\test_stage2_multimodal_diag_floor_adapter.py`：15项PASS。
- `conda activate ssr-gpu; python -m pytest -q tests\test_stage2_multimodal_diag_floor_adapter.py tests\test_stage2_multimodal_concat_fusion.py`：29项PASS。
- 独立review发现并修复：重复append重置step/忽略既有new竞争、full-batch epoch tier漏报、把old raw-score冻结误当无遗忘、query FP64中间与临时内存漏报。
- 核心实现对应本追溯表14项严格设计要求。N607 v2验证C3A/B/C的参数、epoch、step、状态、MAC、FP32和query隔离均PASS，但三者new pooled floor仍为0，old-support非退化仅0/15、2/15、1/15fold通过。
- 全部C3 fold的Stage2-B loss下降且`gamma_abs_max`都达到0.35裁剪上限；C3B/C即使消除cosine distance<0.05的原型中心碰撞，`cls_09f8/cls_f608`仍为0–3.33%。这证明共享对角中心几何不足以解决逐样本类内覆盖和新旧score竞争，核心不得晋级。
- 追溯状态：14/14项verified；“实现符合设计”与“路线性能通过”分开表述，后者为FAIL并已由runner回退C0。
