# 新类注册方法下载与复现验收

用户授权同时做原论文复现和 CVS 场景对比；原数据不能下载时，先在真实 WiSig 上保留作者逻辑进行源代码验收。两种口径分别记录。

## 已有复现

CSIL、MoPC-HR 的 CVS 适配已有完整历史结果，位置：`automation_reports/CV-SincNet/adv3b02_csil_mopc_cvs_adapter_opt_20260724_v3/retrieved/analysis_v1/`。`artifact_audit.json` 记录 400 个完成 cell、400 份预测、400 份评分、1200 个场景行，失败 0。该历史闭合不等于当前 checkpoint 数据来源已复核，也不等于原论文 ADS-B 基准已完整复现；不重复训练已有 run。

## 固定源代码

|方法|仓库|commit|本地只读源|
|---|---|---|---|
|CSIL|https://github.com/pcwhy/CSIL|8ce8637daf4dc60eeb1c56bff64c050c5b2353e9|`local_artifacts/external_refs/pcwhy_CSIL`|
|LoRa_RFFI 作者代码|https://github.com/gxhen/LoRa_RFFI|00948769c05487315b88b7edf40d903fcf8e5b7d|`local_artifacts/external_refs/LoRa_RFFI_20260930`|
|ISSL 作者代码|https://github.com/Simple-up/ISSL|f58eda144063c7f150fbb7c0d0f31aaccb9d3bbe|`local_artifacts/external_refs/ISSL_20260930`|
|LoRa 第三方 Torch 端|https://github.com/PoisonT2/RFFI_Torch|cc6e0e68c52d06869c8382cec0684a4cf651ee76|`local_artifacts/external_refs/LoRa_RFFI_Torch_20260930`|

MoPC-HR 已有作者源：https://github.com/xmuLdz/MoPC-HR ，本地 `github_publish/CVS-RFFI-repo/local_artifacts/external_refs/MoPC-HR`，已读回固定 commit ae6554316ad1a2175920e330133a2f103408bf78。原始仓库保持完整，不往其中写补丁。ISSL 没有许可证声明；本交付提交运行器及源引用，不重发作者源。

LoRa 论文预印本下载： https://arxiv.org/pdf/2107.02867 ，本地 `local_artifacts/newclass_registration_20260930/papers/LoRa_RFFI.pdf`，PDF 12 页已解析核实。
ISSL DOI https://doi.org/10.1109/TCCN.2025.3583118 ，解析到 IEEE document/11050930；未取得全文 PDF。作者仓库 README 只有方法名，无原始数据链接；入口引用 `Dataset_50` 与“徐云公开数据集”，不能据此确认数据版本/切分。LoRa 原数据 https://ieee-dataport.org/open-access/lorarffidataset 页面列出 15.55 GB `LoRa_RFFI.zip`，下载入口要求账户访问。原数据到位前，不声明原论文数值已复现。

## ISSL 实际功能与修改边界

`acceptance.py` 直接加载固定作者 `resnet.py`、`loss.py`；使用 ResNet18 一维骨干，保留原始默认初始化（作者初始化只匹配 Conv2d/BatchNorm2d）、被注释的 dropout、base Adam 0.03、SSL Adam 1e-5、queue K=1000、tau=5、T=20、动量 0.99、0.5 SSL + 0.5 KD、原始 scheduler、drop_last、重复旧类训练样本。

链路为 base → 扩分类头 → SSL → `main_transfer_learningt.py` 的 CE 迁移学习，另从相同 base 执行 `main_increment.py` 对照分支（Adam 1e-4，0.5 CE + 0.5 KD）。不把 SSL 的 6 类头再次扩成 9 类。

必要运行修复：SSL 的 `increment_classes` 返回 tuple，要解包；去除不可达/未定义 `k_num` 的可选保存分支；显式保存本次从零训练的 state_dict；提供独占输出目录与参数化路径，避免旧固定目录覆盖；分类头放到当前 device。WiSig 256 点不满足 4096 crop，诊断按原 4096/4800 比例取 218 点，不插值、不拼包。

**保留并显式验收的上游行为**：KD 返回新的 detached leaf，学生网络无 KD 梯度；SSL 缺少 zero_grad，梯度逐步累积；queue 用 reshape 而非 transpose；初始 queue 未归一化；key 网络的 BN/训练模式保留；扩头的 kaiming 调用原本是 no-op。数值/逻辑问题不因“能运行”而变正确，本变体只证明作者逻辑可执行。若后续按论文修复这些算法问题，必须单独命名变体并取得全文依据，不混入本次代码保留变体。

## LoRa WiSig 端

第三方 Torch 模型结构已逐层核对作者结构：SAME 卷积、残差块、NHWC flatten、512 维嵌入、triplet margin 0.1、KNN 15。运行器补回作者 Glorot/zero 初始化与 RMSprop rho=0.9、epsilon=1e-7。Torch RMSprop epsilon 放置、L2 epsilon 和随机源仍与 TF2.1 不同，未证明完整训练数值等价，不叫作者原版复现。

8192 点 IQ 的作者预处理逐样本对照测试；WiSig 256 点采用 64/32 的 STFT 窗/重叠，保持相邻帧复数比、log 功率和频段裁剪，输出 26×6×1。原版 256/128 对 WiSig 只产生 1 帧，没有可用相邻帧比，运行器拒绝空张量。AWGN 20 至 79 dB、三元组采样（可同样本正对）和 15 邻居分类保留。验证为固定短预算，不实现论文 1000 epoch/早停模型选取；登记不冒充原版完整训练。

## 真实 WiSig 验收

从 N607 ManySig.pkl 只读取数据：6 个 TX、RX 1-1、2021_03_01、raw eq0；每类前 128 条训练、随后 64 条测试，物理 ID 列表和 split 固定存于数据旁 JSON；总训练 768、测试 384；旧 3 类、新 3 类。ISSL 的旧样本重复仅发生在 SSL train 集中，测试不进入任何训练阶段。该切分是执行诊断，不是正式 CVS capsule，不带 LEO，不用于推广/选模或论文准确率宣称。

默认每阶段 1 epoch，仅证明完整训练/预测/保存功能；原论文预算模板是 base=300、SSL=2000、downstream=300，不能靠更改预算冒称原论文数据复现。`acceptance.py` 不读取测试 truth，先固定带物理 ID 的 predictions.npz，再用独立 `score.py` 评分。原方法没有旧类专属适应阶段时 B=N/A。详细 loss 分量、权重、LR、梯度、queue 与方法状态保存在文本 stdout、steps.jsonl、steps.csv；epoch 汇总也在 JSONL。无源验证集时为 null 并说明，绝不取 query 选模。

```powershell
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 paper_reproduction/newclass_registration_20260930/prepare_sources.py
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 paper_reproduction/newclass_registration_20260930/test_runtime.py
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 paper_reproduction/newclass_registration_20260930/fetch_wisig_diagnostic.py --output local_artifacts/newclass_registration_20260930/wisig_diagnostic.npz
# output 必须不存在；正式启动命令、预算和结果见各 experiment.json/report.md。
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 paper_reproduction/newclass_registration_20260930/acceptance.py --method issl --data local_artifacts/newclass_registration_20260930/wisig_diagnostic.npz --output local_artifacts/newclass_registration_20260930/issl_r01
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 paper_reproduction/newclass_registration_20260930/score.py --data local_artifacts/newclass_registration_20260930/wisig_diagnostic.npz --output local_artifacts/newclass_registration_20260930/issl_r01
```

待完成：原数据/PDF补齐、原论文完整预算；正式 CVS 场景矩阵与 LEO/support 协议接入；任何按论文修复的算法变体。验收通过不能替代这些事项。

## 本次已完成结果

2026-09-30 真实 WiSig 验收 **VERIFIED**：ISSL 33 个训练步、384 条预测、4 个 checkpoint strict 重载；LoRa Torch 适配端 12 个训练步、384 条预测、1 个 checkpoint strict 重载。全部逐步 loss/梯度有限，输出与物理 ID 完整。运行设备 RTX 5070 Ti；端到端方法时间 ISSL 21.6653 秒、LoRa 24.6021 秒；Torch peak allocated 520555520/43555328 字节。源代码固定提交 f2aa38aef609d9e8e15f2dfff44e090b778b1852。

|方法|旧类 A|旧类 B|注册后旧类 C|注册后新类 C|H|
|---|---|---|---|---|---|
|ISSL 保留作者行为|33.3333%|N/A|33.3333%|0%|0%|
|LoRa WiSig Torch 端|97.9167%|97.9167%|97.3958%|96.3542%|96.8722%|

两者每阶段只运行 1 epoch，old=3/new=3/K=128，同 RX 同日，不带 LEO；表格只附于运行诊断，不能作为论文准确率对照或正式 CVS 排名。ISSL 低准确率说明短预算未形成有效性能证据，未据此调参/重跑。原代码保留变体与未来论文修复变体不得混用。

[ISSL 完整报告](../../automation_reports/CV-SincNet/20260930-diagnostic-issl-wisig-s392005-r01/report.md) · [LoRa 完整报告](../../automation_reports/CV-SincNet/20260930-diagnostic-lora-wisig-s392005-r01/report.md)。两份报告的 evidence 保存完整紧凑日志、实际参数、数据角色/物理 ID、独立评分与 artifact 审计；大体积权重、IQ、预测留在 `local_artifacts/newclass_registration_20260930/`。

## ISSL修复版（2026-09-30）

原版行为验收入口保留；修复KD计算图、SSL清梯度和queue等问题的入口见[ISSL_REPAIR.md](ISSL_REPAIR.md)。35项修复/协议测试及两组真实WiSig验证通过；长预算主分支旧/新均100%，短预算新类仍0。详细预算、限制及不可覆盖配置见修复报告，不能与此前保留原作者缺陷的结果混称。
