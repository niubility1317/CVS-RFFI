# Task3：平衡split-relative source proxy episodes

## 状态

`LOCAL_VERIFIED`。本任务只实现Phase1 source proxy episode入口；不涉及loss、classification head、训练循环、校准器或target评测，因此不构成任何开放集性能结论。

## 实现范围

| 需求 | 实现与证据 |
|---|---|
|获批split入口|`train_l`映射`proxy_train/L_s`，`val_cal`映射`P_cal/V_cal`，`val_select`映射`P_select/V_select`；每次均调用既有`Phase1DataPolicy.require_proxy_origin`。`train_u`、target和未知角色均抛出`ProxyProtocolError`。|
|周期平衡|`sha256(f"{seed}:proxy")`确定cycle offset；每个完整cycle内每个连续类别ID恰好一次成为proxy。|
|registered隔离|返回`registered_class_mask`、`registered_rows`和`proxy_rows`；proxy类先从mask移除，且两种row集合互斥并覆盖该batch。|
|匿名收据|`schedule_receipt`只含类别数、proxy/registered行数、split role、seed和episode index；不含raw class ID或名称，且为只读映射。|
|fail-closed输入|拒绝空标签、少于3类、负ID、非连续ID、非一维标签、非整数标签、无效seed/episode类型和负episode index。|

## 测试驱动证据

- RED：`conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_proxy.py -q`在新增测试、未创建模块时失败17项，原因是缺少`cvsrffi.phase1_mirage.proxy`。
- GREEN：创建最小`proxy.py`后，相同命令通过17项。
- 限定回归：`test_protocol_policy.py`、`test_data.py`和`test_proxy.py`合计39项通过。

## 环境与版本边界

测试解释器为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，`CONDA_PREFIX=C:\Users\lh594\.conda\envs\ssr-gpu`。本任务未修改`protocol.py`、`data.py`或未跟踪工件；可追溯记录只更新`T3-PROXY-01`行。

## 自审结论

实现没有复制权限表或开放`U_s`真值访问；角色映射只适配公开split名称到既有policy定义。代理类别不会出现在registered mask或registered rows中。标签置换测试验证完整cycle内每个物理row恰好一次承担proxy角色，且所有receipt保持相同的身份无关字段。范围止于episode构造，后续loss/head必须在独立任务中消费该输出。
