# 完整 LoRa 三种子自适应 Tri 与 practical 一致性

## 同预算矩阵

数据为原注册 LoRa 7% 标注合同，L=2800，U=25200，V=12000，完整目标 15 个未见 RX × 2000 物理包。三种子为 2026092701、2026092702、2026092703。每种子内四行都从 `20261122-phase1-lora-full-supervised-m6-r01` 中独立冷启动的 7392 步监督权重出发，固定再训练 28 轮×88 步（2464 步），末轮冻结。保留已配对修复小试的 clean CE、0.35 practical CE、按 L 起点召回加权、AdamW/余弦预算，以及同一物理 L、同 augmentation 场景与独立 RNG。三条半监督行对 U 每轮从同一盲排列无放回取 2800 个候选，每步取与 L 同数，U 采样与扰动使用不同 RNG。U 的 TX 真值仅在预测锁定后独立读取。

| 方法 | 固定监督之外的差异 |
|---|---|
| continued_sup | 继续纯监督 |
| plain_pl | 普通在线 0.95 硬 PL |
| adaptive_tri | 完整留源规则与本 seed 教师 V 的交集，至少两个跨 RX 可靠类，按支持数给硬 CE 权重 max 0.40，逐 batch 每类 cap 3。使用真 L 同类异 RX 供体，以及正交奇偶 chirp 的低剂量同包一致性 |
| tri_practical | 保持 adaptive_tri 的教师、硬准入、跨 RX 配对和损失系数，只在同包一致性项把交替的一半候选替换为对应的源 practical 视图，去啁啾相干度弱于 0.40 的半数仍用原始输入，始终不向 practical 视图赋予硬 TX |

单种子开发对照的 Tri 在源 practical V 比同预算普通 PL 低 0.31 个百分点。新 practical 行以这条源证据为动机，隔离一致性的输入变化。全部 12 行都完成七种目标读出，无条件删行或用目标挑配方。对比每种子相对于同 seed 普通 PL 和继续监督的净纠错，报告 clean、逐场景与六景均值、三种子波动、逐 TX 和 RX、最差 RX、U 支持和准确率。比较同优化步数，教师缓存和同包第三视图的额外计算单列。

## 发布与停止规则

`20261122-phase1-lora-full-aligned-data-r01` 生成原 1 MHz 上正确同步的 native/waveform 视图，监督师与留源折均冻结后才允许配置每个 seed 的门控。发布 `python -m experiments.pamf_lora_main.full_adaptive_publish --repo REPO --output OUT --context CONTEXT --ssh-config SSH_CONFIG --data-submit DATA_RECEIPT --supervised-submit SUP_RECEIPT`。在 N607 普通 CVS-RFFI 环境的 `CV-SincNet/releases/20261122-phase1-lora-full-adaptive-m12-r01-release` 启动，输出 `CV-SincNet/runs/20261122-phase1-lora-full-adaptive-m12-r01`。GPU0–6 按真实容量，每卡本次一个进程与全卡至多四个，跳过 GPU7。失败的前置状态、源监督权重不匹配、U 标签/物理顺序错误、非有限值、预测行数缺失停止后续派发并保留文件。源训练进程不接收目标路径。真值侧 `python -m experiments.pamf_lora_main.full_adaptive_score` 等 12×7×15 份盲预测全部结束，逐份核对长度与有效类别后再跨 30000 物理条绑定官方标签，最后做独立 U 真值审计。

代码能够记录每轮完整 L/V 节点和实际 U 相干度替换量。使用已有同模型结构的真实权重和源 IQ，本地复算了末批 16 帧与四个 optimizer 分支。主 3 种子真实权重 smoke 由依赖训练结束后自动执行，终止行为按同一 release 固定。
