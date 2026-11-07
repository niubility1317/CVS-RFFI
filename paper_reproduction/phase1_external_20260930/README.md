# Phase1 三项 WiSig 对比方法

2026-09-30：ASKNet、域不变特征学习方法和 WaveMLP 已下载并修复，均通过合成数据的原生训练、保存/读回和测试检查。按用户要求保留作者主要逻辑、功能和分类结构，没有改成另一套论文重写实现。

| 方法 | 作者来源 | 固定提交 |
|---|---|---|
| ASKNet | [ASKNet-RFF](https://github.com/BeechburgPieStar/ASKNet-RFF) | `f00846d68425315b68ae69b8c6b712e08559c2fe` |
| 域不变特征学习（内部简称 DIFL） | [Receiver-agnostic-RFFI-CL-](https://github.com/Edith-xx/Receiver-agnostic-RFFI-CL-) | `feae373ccc59bc427c0632d0945f0a268172227e` |
| WaveMLP | [WaveMLP-RFF](https://github.com/BeechburgPieStar/WaveMLP-RFF) | `1ade72d0082428db2087633a44ae533f1022ca22` |

三者状态均为 `REPAIRED_SYNTHETIC_VALIDATED_PENDING_MATCHED_DATA`。详见 [V2 修复记录](REPAIRS_V2.md)、[验证结果](repair_results_v2.json) 和 [方法清单](methods.json)。[V1 检查记录](AUDIT_V1.md) 保留原缺陷和历史证据。

工作代码：`E:/type10-7/code/external_baselines/phase1_20260930/{asknet,difl,wavemlp}/`。不可变作者克隆、原权重及历史文件：`E:/type10-7/local_artifacts/phase1_external_20260930/upstream/`。未加载作者权重。固定提交与远端读回一致，下载状态 VERIFIED。

版本化交付采用固定来源、精确补丁和验证工具，不重新发布第三方完整源码。DIFL 作者 README 标注非商业许可；ASKNet/WaveMLP 固定树未发现独立许可证。

从本目录运行下面命令可准备新副本并执行合成检查。使用 ssr-gpu Python；大写路径替换为不存在的新目录。工具拒绝覆盖已有目录。

```text
python -X utf8 prepare_sources.py --upstream-root E:/type10-7/local_artifacts/phase1_external_20260930/upstream --working-root NEW_WORKING_COPY
python -X utf8 repair_validation.py --upstream-root E:/type10-7/local_artifacts/phase1_external_20260930/upstream --working-root NEW_WORKING_COPY --output-dir NEW_EVIDENCE_DIR
```

默认应用 V2；`--repair-version 1` 仅用于重建历史状态。`audit_smoke.py` 是 V1 缺陷复现，V2 使用 `repair_validation.py`。

此次未启动真实 WiSig/N607 实验。正式公平对比还需接入冻结物理样本划分、少标签权限及实验登记。合成检查不证明论文准确率或完全论文一致性。
