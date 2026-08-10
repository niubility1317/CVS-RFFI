# Phase1单读出local4控制bundle v1：v5本地native隔离追溯

状态：`LOCAL_NATIVE_ISOLATION_VERIFIED / INDEPENDENT_REVIEW_P0_0_P1_0_ALLOW / NO_REAL_BUILD_RESULT`。

## 固定边界

v5只修复v3/v4真实ManySig构建中长期单进程native状态累积和隐藏双PKL读取风险。它不改变F1C输入、L/U/V索引、view seed、三种LEO场景、geometry、五维`domain_descriptor`、median/MAD、tail、TorchScript运行时、10成员bundle schema、resource gate或CARE语义。v4的唯一真实run已由`build.exit=139`和`Segmentation fault`证实为技术停止；本卡不把该现象归因为OOM，也不宣称v5已完成真实构建。

## 追溯表

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|SCB-V5-01|v4终态、严格loader合同|ManySig只执行一次`pickle.load`，直接完成dict/data/TX/RX/day/equalized严格校验，不再调用历史loader，不再在source helper重复计算dataset SHA|`code/cvsrffi/phase1_single_control_bundle_v1.py`、测试|verified|真实小PKL的`pickle.load==1`且历史loader不可达|不改变数据内容或split|
|SCB-V5-02|v4 SIGSEGV、冻结数学定义|L/U descriptor以及L/V runtime、V descriptor均按固定工程chunk由全新Python解释器执行；worker使用固定单线程native环境|core、测试|verified|至少两个真实chunk与单进程reference逐字节/`np.array_equal`一致|chunk大小固定为`512`，不是CLI或方法超参|
|SCB-V5-03|数据与身份边界|U worker仅接受opaque index、必要raw IQ bytes和固定view信息，只返回descriptor数值；label、physical key及其hash出现即拒绝|core、测试|verified|输入和输出负测均fail-closed|临时stdin/stdout不写入bundle、跨run缓存或报告|
|SCB-V5-04|输出根与fail-closed合同|worker非零、signal/139、schema、行数或顺序异常时一次调用即停止；不得进入`build_bundle`，output/staging保持ABSENT|core、测试|verified|exit139、乱序、缺行和真实builder接线负测；错误含role/chunk/signal|不重试，不输出IQ或feature|
|SCB-V5-05|生命周期与既有parity合同|主进程在source循环期间释放eager/loaded runtime；结束后短生命周期重建parity/smoke，receipt数值及bundle成员不变|core、测试|verified|既有fixture 10成员/schema/parity/resource/CARE回归通过|不修改`np.fft`、trace、median/MAD或resource门|

## 本地验证与冻结文件

`ssr-gpu`下已串行通过`python -m py_compile code/cvsrffi/phase1_single_control_bundle_v1.py code/scripts/build_phase1_single_control_bundle_v1.py code/tests/test_phase1_single_control_bundle_v1.py`、`pytest -q code/tests/test_phase1_single_control_bundle_v1.py`（`37 passed`）和公开build CLI`--help`。`git diff --check`通过。当前实现字节SHA256：`code/cvsrffi/phase1_single_control_bundle_v1.py=1d36b95beeae9044d44b311c982deddb9ad7a0f6b5112094486a28c51c80ead1`；`code/tests/test_phase1_single_control_bundle_v1.py=99e61e6e294f5802cb4f1e5d69d600e679eae080cdf7da1d48f6880e9cd57a15`。

## 验收限制

本轮局部测试与独立复审`P0=0／P1=0／ALLOW`只证明实现路径和闭合负测。真实F1C＋ManySig构建和任何Phase1或性能结论均不在本轮完成范围内。
