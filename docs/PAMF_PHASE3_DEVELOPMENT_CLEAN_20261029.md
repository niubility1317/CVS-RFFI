# Phase3 RX24-1 clean 配对开发控制

本控制保留已完成 RX24-1 星地开发池的全部 1248 个原始物理记录与 opaque ID 顺序，生成独立的 clean 公共输入。U_dev 为 384 条，V_dev 为 864 条；不重新抽样、不按质量筛选、不生成信道，不读取正式 RX24-18。它用于区分跨接收机 clean 与叠加星地后的变化，不增加独立物理样本数。

用户当前科学目标为跨 RX clean 未知拒识≥95%、跨 RX 星地未知拒识≥90%；继续要求 known FRR≤5%、相对原 F 同输入开放准确率损伤≤2 个百分点，并在约束内提高 known 接收率。本文件只交付数据控制能力，不报告这些指标已达成，也不授权模型使用真值或 clean 恢复参考。

## 固定输入与构建方式

配置 [pamf_phase3_development_clean_20261029.json](E:/type10-7/configs/pamf_phase3_development_clean_20261029.json) 只接受 `full_root`、`paired_pool` 两个路径。`paired_pool/complete.json` 必须是 `PHASE3_DEVELOPMENT_POOL_VALIDATED / VALIDATED_ONCE`，其 builder report 必须保持既定 RX24-1、日期、类别与角色。builder 只读取已有 `mapping` 中的记录，按 public 已混洗次序逐条核对 opaque ID、canonical record 和两份已索引 source archive/member。

原 Full 158 个成员已经验证，本控制复用父报告的 `validation_reused`，不扫 ZIP 目录、不重验原数据或增设 hash／receipt。只打开 RX24-1 在 `2021_03_08` 和 `2021_03_15` 的两个既定 NumPy pickle 成员；原始 packet index 按父映射直接定位，不重新调用选择器。仅对选中原记录调用一次原有 `experiments.phase1_external_dg.data.iq_record`，按 FP32 转置及 unit RMS 形成 `[2,256]`，不再均衡、不生成卫星观测，也不读取父池的 received IQ。

CLI 首行报告实际 PID、CWD 和 argv。输出 root 不可覆盖，两成员与映射检查完成后才创建输出，所有公共文件与私有报告检查通过后最后写 `complete.json`。入口如下；本次实现交付不执行真实构建。

```text
python -B -u -m experiments.pamf_phase3_open_world.development_clean --config configs/pamf_phase3_development_clean_20261029.json --output local_artifacts/pamf_phase3_development_clean_20261029_r01/pool
```

## 公共接口与条件隔离

每个 role 的 `public/` 严格只有 `manifest.json`、`ids.npy`、`received.npy`。manifest 保持旧六类顺序 `14-10,14-7,20-15,20-19,6-15,8-20`，不公开新类 TX、私有 truth、canonical mapping 或父输入路径。clean 与 star 的 opaque IDs 同序相等，`capsule_id` 和 `split_id` 则分别标识不同观测条件。

专用 API：

```python
from experiments.pamf_phase3_open_world.development_clean import load_clean_public
manifest, ids, received = load_clean_public(clean_public_root)
```

返回 manifest、Unicode opaque IDs 和只读 FP32 `[N,2,256]` memmap。API 验证精确字段／文件白名单、六类身份、形状、数量、唯一 ID 及有限数值。schema 为 `phase3_clean_min_v1`，`processing_route='clean'`、`mode='clean'`。既有 `data.load_public` 的 `phase3_min_v1 / residual / post_sync` 合约保持原样，并拒绝此 clean 输入。

`single_observation=true` 只表示 clean 条件内每条原记录有一份固定观测；与父 star 配对的两种条件不是两批独立物理样本。clean 与 star 必须作为独立公共输入运行，星地方法不能读取逐条 clean 参考或借 clean 真值拟合恢复器／适应状态。此 builder 没有加载 F 或 G，不建立两条件间的模型侧引用。

builder 私下复制父 truth 的 `ids/y/receiver`，仅将 `scenario` 改为 `clean`；truth 留在每个 role 的 `truth/truth.npz`，模型不能读取。canonical 映射由私有 builder report 引用父 `builder_report.json#mapping`，不复制到 public。报告记录 `source_parent`、两个 archive/pickle reads、1248 个唯一物理记录、零新采样、零信道、零 F/G／模型活动、零正式 RX24-18 访问，以及 `paired_original_not_new_records=true`。

## 本次验证范围

先观察模块缺失导致的 8 个预期失败，再实现 builder／loader。clean 专项最终 **8 tests＋29 subtests PASS**；与既有 `tests/test_pamf_phase3_data.py` 合并的窄范围回归为 **16 tests＋32 subtests PASS**，配置校验与语法解析通过。

只使用合成 Full pickle 和合成 parent metadata／truth，调用真实 `iq_record`；覆盖错 parent／IDs／canonical／成员、无信道与无重抽样、只读两个映射成员、私有文件不公开、loader 契约和数值异常、原 root 防覆盖与 CLI 首行身份。没有读取真实 IQ 或权重，没有真实 build、模型运行、评分、发布或提交。真实构建与独立 P0/P1 检查由本 run 的 launch owner 按项目最小流程完成，数据准备不等于科学目标完成。

## 实际构建完成（2026-10-29）

run `20261029-phase3-development-clean-rx24-1-r01` 已VERIFIED/VALIDATED_ONCE/ANALYZED_DATA_ONLY，code `959a1a68e76a7e2cfba6700c2a251da8330e59b9`。实际U384/V864，沿用parent同序ID和1248条原物理记录，两个既定Full成员、零新采样/信道/F/G/预测/正式target。实际PID43708的原生命令、CLI CWD及起始live进程已核实，日志541→838B、5行完整读取无异常；完成标记和公共三文件齐全，进程已退出。

专项8 tests/29 subtests及相邻16 tests/32 subtests PASS的原始stdout只留在实现worker工具回合，未保存磁盘测试日志；登记已将误指的focused_tests.log设null，并保留这一观察限制，不为补日志重复测试。一次独立P0/P1 PASS。真实构建日志与post/complete/analysis_summary有独立保存；本构建没有科学性能结果。

|角色|capsule_id|split_id|
|---|---|---|
|discovery|p3dev-clean-ef10219d27274e04e83408d1|64274719133a83590352add5|
|evaluation|p3dev-clean-20e01e9ae54a4bc3ae503a3f|347fd2cb757a37f7f162223a|

真实IQ和truth保持本机local_artifacts，不提交Git；clean与star模型输入隔离，目标95%/90%尚未证明。
