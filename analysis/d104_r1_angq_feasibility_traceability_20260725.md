# D104-R1-ANGQ-RXID-MB4可行性追踪

状态：`DESIGN_FROZEN / IMPLEMENTATION_REPAIRED_PENDING_INDEPENDENT_REREVIEW / N607_NO_GO / TARGET25_NO_GO / NO_PERFORMANCE_RESULT`

根目录正式报告：`E:\type10-7\automation_reports\CV-SincNet\d104_r1_angq_feasibility_20260725\report.md`

报告SHA256：`76dfdc4988de507d18051a56623327527838f0bce51242f017f1662e4bc62f00`

|项目|证据|结论|
|---|---|---|
|设计首版|commit`33f3d350`|独立复审`NO_GO / P0=0/P1=7/P2=2`|
|部署同构修订|commit`2034f724`|已冻结FP16先舍入后编码、ties-to-even、归一化和tie-break语义|
|第二轮审查|commit`35acfc21`|`NO_GO / P0=0/P1=4/P2=3`|
|REV3闭环|commit`ab0ba7f7`|修复wire资源、tie、历史暴露、joint、方向审计和逐位测试|
|第三轮独立复审|HEAD`3419ac20`|`GO / P0=0 / P1=0 / P2=2`；仅授权正式本地实现|
|tap全池性质|8400条；7575改善、825相同、0退化|含2478条历史诊断query；0新held、0truth|
|已知边界修复|K10 `1-1`从298/300到300/300；`2-1`从309/310到310/310|端到端和共享带宽方向审计均0翻转|
|定向验证|13项pytest通过；2脚本py_compile通过；diff check通过|不替代held性能|
|正式资格|第三轮独立复审`P0=0/P1=0`|正式本地实现解锁；N607、正式source-held和Target25仍禁止|
|身份修复re-entry|真实2478-ID canonical root=`7870604d...b558`；旧`036456...854d`不可复算|独立裁决`P0=0/P1=1/P2=1`；split升v2并新增自描述manifest，复审前暂停实现|
|身份修复首轮复审|commit`a54e4284`|`P0=0/P1=2/P2=2`；要求硬锁输入/代码/registry/package身份并修正标签使用声明|
|身份修复补丁|commit`73e4cbd7`；manifest r3；8项pytest＋1项py_compile|硬锁tap/dual/代码/source-val/support/package身份，使用排他写入并拆分标签用途|
|身份修复终审|commit`73e4cbd7`|`P0=0/P1=0/P2=1`；恢复`DESIGN_FROZEN / IMPLEMENTING_LOCAL_ONLY`；N607/source-held/Target仍NO_GO|
|正式实现|commit`2006564f`；104项测试|246-fit、21包、63row、252arm-row、truth-side scorer与gate落地|
|实现首轮release复审|commit`2006564f`|`NO_GO / P0=0/P1=4/P2=3`；未同步、未启动|
|release修复|当前待提交差异；D103/D104相关测试93项通过|补齐truth-before-open内部封印、整数指标/效应重算、split/matrix/scorer/root绑定、首波与runner资源receipt；待独立复审|

D104只改变typed qKNN的逐support量化尺度选择。D103跨receiver MetaBias4、Phase1机制、全类统一Student-t评分、query隔离和资源门均不改变。旧非部署同构r4结果已撤回。

新source split固定排除旧诊断query的2478个物理ID，并以新salt按receiver×TX×day每cell取15条held。容量审计只证明168个cell均可取满，不证明builder正确性或性能。

当前无BA、floor、H、old/new准确率、forgetting或Target结果。正式runtime必须只归一化输入一次，绑定8400行执行c=1 scale/code/decoded逐位回归，并按`32320×registered_class_count×K`发布MAC总量。N607仍有驱动/NVML不匹配，且旧D103量化门已知失败；未同步、未启动、无远端输出。

## 正式实现追踪

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|D104-RT-01|重入卡§2|严格实现101点ANGQ；输入只执行一次`normalize64→32`；FP16 scale、ties-to-even、stable-first与fail-closed语义不漂移|`code/cvsrffi/stage2_d104_angq_qknn.py`|verified_local|定向pytest＋py_compile|禁止按receiver、K、场景、类别或角色改变网格|
|D104-RT-02|重入卡§2|builder没有query参数，保留typed qKNN数值ABI、registry顺序和Student-t全类评分兼容性|`code/cvsrffi/stage2_d104_angq_qknn.py`、`tests/test_stage2_d104_angq_qknn.py`|verified_local|签名/类型/序列化往返/全同分负测|不得修改通用D103 builder默认行为|
|D104-RT-03|重入卡§2、§5|K1沿用`h0`，K5/K10按ANGQ解码support重算FP16类带宽；发布端到端teacher/deployed量化审计|同上|verified_local|K1/K5/K10单元测试|teacher使用FP32 support与FP32带宽|
|D104-RT-04|重入卡§5|发布参数化资源receipt、数值bank数组delta、metadata/wire delta与query MAC delta|同上|verified_local|C、K参数化断言|C=6、K10常数不得外推|
|D104-RT-05|复审P2-1|绑定8400行执行`c=1`的scale/code/decoded逐位等价；正式路径无重复输入归一化|`code/scripts/verify_d104_angq_c1_full_tap.py`、对应测试|verified_local|8400行scale/code/decoded变化0|旧tap含2478条历史诊断query，只用于ABI回归|
|D104-RT-06|项目§5.3–§6|非法dtype/shape/零范数/非有限值/不平衡K/角色或query参数均fail closed|`tests/test_stage2_d104_angq_qknn.py`|verified_local|协议负测|类标签置换等价|
|D104-RT-07|重入卡§3|实现M0/M_DA/M_HEAD/M_JOINT四臂method lock与同row预测闭合|`code/cvsrffi/stage2_d104_rxid_angq.py`|verified_local|四臂单元测试＋真实特征K5烟测|M_DA不得按行回退为晋级臂|
|D104-RT-08|重入卡§4|实现新source split builder与63×4=252个prediction单元runner|builder/runner/validator|verified_local|真实split metadata＋不可覆盖publisher＋252 receipt/scorer/gate测试|首次打开truth前封存全部prediction|
|D104-RT-09|AGENTS release readiness|真实checkpoint no-query smoke、协议负测和独立release复审|测试、报告与release追踪|repaired_pending_rereview|真实K5 400-step烟测、v2 split单fit400-step；首轮复审P0=0/P1=4/P2=3；修复后D103/D104相关测试93项通过|重新提交并复审至P0=0/P1=0前N607保持NO_GO|
|D104-RT-10|AGENTS N607发布|本地Git提交后由专职runner执行N607预检、同步、25矩阵与证据回收|正式run报告|blocked|N607 GPU栈恢复＋runner handoff|当前driver/NVML不匹配；本地实现仍可推进|

## Release修复追踪

|ID|首轮release复审问题|实现闭包|证据|
|---|---|---|---|
|D104-RF-01|truth首次打开前内部预测证据不完整|truth-blind validator重算prediction receipt、四臂内容、共同method lock、row method lock、registry、support/query root、scorer输入封印|内部封印篡改在event/truth读取前失败|
|D104-RF-02|gate信任浮点与硬编码资源|从六类整数重算BA/floor/joint/correct及全部2×2效应；逐row重算资源、INT8、runner收据|`abs_tol=1e-12`；篡改负测|
|D104-RF-03|D104未绑定D103实际矩阵输入与访问闭包|不可覆盖`run_input_binding.json`绑定split、L/U/source-val、历史排除、checkpoint/runtime/method lock、scorer、matrix plan和246-fit access root|root/path/SHA逐字段复核|
|D104-RF-04|split标签用途声明错误|source标签用于receiver×TX×day分层明确为true；方法选择、性能选择、source-val性能均false|真实split r2 metadata SHA=`1bf60470…1e67`；publisher SHA=`4a1e23cc…da21`|
|D104-RF-05|P2路径、负测和首波证据不足|精确21包身份/路径闭包、7类封印负测、首波16-fit、唯一fit ID、matrix plan、access root和runner resource schema|D103/D104相关测试93项通过|
