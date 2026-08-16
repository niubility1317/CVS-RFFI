# Task3：平衡split-relative source proxy episodes

## 状态

`LOCAL_VERIFIED`。本任务只实现Phase1 source proxy episode入口；不涉及loss、classification head、训练循环、校准器或target评测，因此不构成任何开放集性能结论。

## 实现范围

| 需求 | 实现与证据 |
|---|---|
|获批split入口|`train_l`映射`proxy_train/L_s`，`val_cal`映射`P_cal/V_cal`，`val_select`映射`P_select/V_select`；每次均调用既有`Phase1DataPolicy.require_proxy_origin`。`train_u`、target和未知角色均抛出`ProxyProtocolError`。|
|周期平衡|`sha256(f"{seed}:proxy")`确定cycle offset；当前batch按每类首次出现的row位置形成匿名类组顺序，每个完整cycle内每个连续类别ID恰好一次成为proxy。|
|registered隔离|返回`registered_class_mask`、`registered_rows`和`proxy_rows`；proxy类先从mask移除，且两种row集合互斥并覆盖该batch。|
|匿名收据|`schedule_receipt`只含类别数、proxy/registered行数、split role、seed和episode index；不含raw class ID或名称，且为只读映射。|
|fail-closed输入|拒绝空标签、少于3类、负ID、非连续ID、非一维标签、非整数标签、无效seed/episode类型和负episode index。|

## 测试驱动证据

- RED：`conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_proxy.py -q`在新增测试、未创建模块时失败17项，原因是缺少`cvsrffi.phase1_mirage.proxy`。
- GREEN：创建最小`proxy.py`后，相同命令通过17项。
- 初始限定回归：`test_protocol_policy.py`、`test_data.py`和`test_proxy.py`合计39项通过。

## 环境与版本边界

测试解释器为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，`CONDA_PREFIX=C:\Users\lh594\.conda\envs\ssr-gpu`。本任务未修改`protocol.py`、`data.py`或未跟踪工件；可追溯记录只更新`T3-PROXY-01`行。

## 自审结论

实现没有复制权限表或开放`U_s`真值访问；角色映射只适配公开split名称到既有policy定义。代理类别不会出现在registered mask或registered rows中。标签置换测试覆盖完整cycle角色平衡与同一episode的row集合、映射后类别和receipt一致性。范围止于episode构造，后续loss/head必须在独立任务中消费该输出。

## Fix round1：I1逐episode标签置换等价

原实现按raw class ID排序；标签置换后，固定seed与episode可能选中不同物理类。使用`[0,1,1,2,2,2]`和双射`0→2、1→0、2→1`的RED测试复现：原batch的proxy row为`[0]`，置换batch为`[1,2]`，同时receipt行数改变。

修复后，`proxy_class_for_episode`保留其既有签名，但将输入`Sequence`解释为匿名first-seen类组顺序，并以保序去重替代数值排序。`build_proxy_episode`已把原始labels顺序传入该函数；标签双射不改变各物理类的row集合或首次出现位置，因此同一seed与episode返回相同`proxy_rows`、按双射对应的`proxy_class`和相同receipt。

- I1 RED：新增不均衡类别的同一episode测试后，`tests/phase1_mirage/test_proxy.py`失败1项，断言显示`[0]`与`[1,2]`不一致。
- I1 GREEN：同一proxy测试文件18项通过。
- I1限定回归：`test_protocol_policy.py`、`test_data.py`和`test_proxy.py`合计40项通过。
