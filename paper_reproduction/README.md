# 论文原始基线复现目录

本目录只承载论文原始设定复现，不接入CVSStage2-C、OA-MSE、`z_id/z_dom`、unknown gate或项目扩展评估口径。

## 已落地基线

- `protonet_cda`：`Cross-Domain Adaptation for RF Fingerprinting Using Prototypical Networks`的目标域K-shot support原型计算、距离softmax和query NLL/CE核心。
- `feature_separation_crossrx`：`Few-shot Cross-Receiver Radio Frequency Fingerprinting Identification Based on Feature Separation`的共享编码器、transmitter分支、receiver分支、TX/RX分类损失和分支相关性惩罚核心。

## 本地证据路径

- ProtoNet论文PDF：`E:\type10-7\RFFI少样本学习\联网扩展_20260624\RFFI方向_论文\02_跨接收机_域适应_域泛化_信道鲁棒\RFFI-DOM-10_Cross-Domain_Adaptation_for_RF_Fingerprinting_Using_Prototypical_Networks.pdf`
- Feature Separation论文PDF：`E:\type10-7\RFFI少样本学习\联网扩展_20260624\RFFI方向_论文\02_跨接收机_域适应_域泛化_信道鲁棒\AUTH-RFFI-X02_Few-shot_cross-receiver_radio_frequency_fingerprinting.pdf`
- 综述抽取文本：`E:\type10-7\tmp\abce_docx_extracted_20260625.txt`

## 当前边界

训练入口已接入为配置校验和dry-run护栏；formal模式会阻断`paper-unspecified`和未解析占位字段。真实WiSig/目标接收机训练仍需在配置中填入具体receiver/day、N-way/K-shot、λ、epoch和seed后再启动。凡论文原文细节尚未逐项确认者，统一记录在`repro_gap.md`，不得在结果表中冒充论文设定。
