# D104-R1-ANGQ-RXID-MB4可行性追踪

状态：`DESIGN_FROZEN / IMPLEMENTING_LOCAL_ONLY / N607_NO_GO / TARGET25_NO_GO / NO_PERFORMANCE_RESULT`

根目录正式报告：`E:\type10-7\automation_reports\CV-SincNet\d104_r1_angq_feasibility_20260725\report.md`

报告SHA256：`44e836a30cf052fc16baf0d5b130f5c86bfd5e0aa3eb5143cccccbb419e4040f`

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

D104只改变typed qKNN的逐support量化尺度选择。D103跨receiver MetaBias4、Phase1机制、全类统一Student-t评分、query隔离和资源门均不改变。旧非部署同构r4结果已撤回。

新source split固定排除旧诊断query的2478个物理ID，并以新salt按receiver×TX×day每cell取15条held。容量审计只证明168个cell均可取满，不证明builder正确性或性能。

当前无BA、floor、H、old/new准确率、forgetting或Target结果。正式runtime必须只归一化输入一次，绑定8400行执行c=1 scale/code/decoded逐位回归，并按`32320×registered_class_count×K`发布MAC总量。N607仍有驱动/NVML不匹配，且旧D103量化门已知失败；未同步、未启动、无远端输出。
