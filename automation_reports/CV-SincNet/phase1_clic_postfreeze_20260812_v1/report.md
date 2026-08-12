# Phase1 CLIC后冻结v1预注册与运行报告

## 1. 状态与目标

- 实验ID：`phase1_clic_postfreeze_20260812_v1`
- 当前状态：`LOCAL_VERIFIED / SOURCE_FIRST_WAVE_PENDING`
- 操作者：主控Codex；N607唯一runner：`Luna/max`
- 训练输入：不可覆盖run`phase1_clic12_20260812_v5`的12个final checkpoint与12个terminal receipt。
- 目标：先执行12个source-only clean／fixed400 proxy导出，再用每fold一次预固定、C／G共享字节的三场景source-L LEO弱信道缓存完成LEO／pair／bundle；目标域阶段只在IQ-only隔离修复和既有`p2_min_v1`目标包原件闭合后执行。

## 2. 科学合同

- G只从同一份`received_i`提取固定lag=`{1,2,4,8}`的多尺度三点复曲率token；不增加观测、view、物理样本或K。
- 每项正式结果都必须覆盖`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；目标同时报告registered-known域泛化和registered／unknown开放集拒识。
- source clean／proxy阶段只读ManySig、v5 checkpoint和terminal；target／query／truth／role访问为0。
- source LEO缓存按fold生成一次，C／G共同只读同一NPZ；每个物理样本仅分配一个scene，不择优、不融合、不重采样。
- ADV3B02只要求训练数据配置、known-test配置、信道和指标口径等价，不要求与CLIC同capsule、同物理样本或同received-IQ字节。

## 3. 第一波冻结矩阵

|阶段|行数|输入|输出|GPU|
|---|---:|---|---|---|
|source clean／proxy export|12|F1—F6×C/G checkpoint、terminal、ManySig|每臂一个`source_clean_proxy.npz`|0—7按训练矩阵固定映射|

第一波不运行source LEO、pair、bundle、target预测或truth-side scorer，不读取性能。它是完整60步矩阵的前12个source clean步骤，不改变后续步骤数。

## 4. 版本与本地验证

- 当前工作分支：`codex/phase3-responsibility-20260807`。
- 训练完成报告commit：`d7887f0a`。
- 后冻结真实绑定：training=`phase1_clic12_20260812_v5`，postfreeze=`phase1_clic_postfreeze_20260812_v1`。
- `ssr-gpu`验证：六个生产／测试模块`py_compile`通过；`code/tests/test_phase1_clic_postfreeze.py`为`100 passed`，仅3条既有AMP弃用warning。
- ManySig路径：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`；SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。

## 5. N607路径与启动合同

- 训练根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic12_20260812_v5`。
- 后冻结输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_postfreeze_20260812_v1`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_postfreeze_20260812_v1`。
- 第一波每行只运行`export_phase1_clic_features.py`，要求新输出路径不存在；禁止覆盖或恢复。
- 每行预期工件：`source_clean_proxy.npz`与独立日志；12／12成功前不解释性能。
- 技术停止条件：错误checkout／hash、输出覆盖风险、协议字段漂移，或至少两个独立行在输出前出现相同确定性异常。不得按accuracy、AUROC、loss或其他性能停止。
- retry：`NO`。若系统性技术故障，封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，本run不得重启。

## 6. 当前已知阻塞与分阶段策略

- 现成source-L LEO缓存没有一个已证实同时包含v5每fold所需的TX／RX／day／physical ID／三scene／received-IQ字段；因此先执行不依赖它的12个clean／proxy导出，再从v5 source-L物理划分一次性生成共享缓存。
- target实现独立复审发现IQ-only package可能在模型forward前暴露known-test配置路径；目标阶段必须先修为模型只见IQ／opaque token／scene，全部forward结束并销毁runtime后才由离线封存器追加配置绑定。该问题不影响source第一波。
- 当前状态不是性能结果，不进行方法晋级或淘汰。
