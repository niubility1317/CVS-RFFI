# PAMF 与固定 DURA v1.0＋PAMF 目标测试结果

同一固定 unseen-RX 测试集，每景 168000 条物理样本，三个模型种子。表内为准确率均值 ± 样本 SD（%）。两个 readout 分别对同 readout 的监督 F20、普通 PL F21 比较。

PAMF 单模型

| 视图 | F20 监督 | F21 普通 PL | HPL_E20_AS_CAP2 |
|---|---:|---:|---:|
| clean | 84.785317 ± 0.823592 | 84.981746 ± 0.189284 | 84.461111 ± 0.207458 |
| practical_high | 80.043254 ± 1.106004 | 79.775992 ± 1.199662 | 81.258929 ± 0.612792 |
| practical_mid | 77.570040 ± 1.099521 | 77.280357 ± 1.103033 | 79.108730 ± 0.511207 |
| practical_low_suburban | 75.079167 ± 0.976186 | 74.836706 ± 0.970531 | 76.875992 ± 0.410488 |
| practical_high_urban | 76.625000 ± 1.091559 | 76.381151 ± 1.106568 | 78.084524 ± 0.575164 |
| practical_mid_urban | 62.620040 ± 0.918716 | 62.403373 ± 0.894281 | 64.876190 ± 0.471399 |
| practical_low_urban | 51.639683 ± 0.549972 | 51.478571 ± 0.525797 | 54.293452 ± 0.377889 |
| six_mean | 70.596197 ± 0.951923 | 70.359358 ± 0.957619 | 72.416303 ± 0.443835 |

固定 DURA v1.0＋同一 PAMF

| 视图 | F20 监督 | F21 普通 PL | HPL_E20_AS_CAP2 |
|---|---:|---:|---:|
| clean | 84.857540 ± 0.869438 | 85.029563 ± 0.204130 | 84.551190 ± 0.202992 |
| practical_high | 82.592460 ± 0.587476 | 82.821429 ± 0.210169 | 81.739683 ± 0.534902 |
| practical_mid | 81.282341 ± 0.500683 | 81.471032 ± 0.176189 | 80.570040 ± 0.635605 |
| practical_low_suburban | 79.645833 ± 0.379969 | 79.815278 ± 0.181221 | 78.910913 ± 0.764269 |
| practical_high_urban | 81.353175 ± 0.576988 | 81.602183 ± 0.225621 | 80.547421 ± 0.575978 |
| practical_mid_urban | 74.553175 ± 0.231744 | 74.798214 ± 0.249648 | 73.662698 ± 0.788951 |
| practical_low_urban | 67.702381 ± 0.203648 | 67.939683 ± 0.343248 | 66.955159 ± 0.775990 |
| six_mean | 77.854894 ± 0.396039 | 78.074636 ± 0.184312 | 77.064319 ± 0.677872 |

原始研发目标继续按 PAMF 单模型的整数正确数验收：对两个对照，clean 均至少 +1 pp，六景平均均严格超过 +1.5 pp。DURA 组合为并列参考，不替代单模型验收。

HPL_E20_AS_CAP2 / raw：未同时满足数值门槛。
对 F20：clean -0.324206 pp；六景 +1.820106 pp。
对 F21：clean -0.520635 pp；六景 +2.056944 pp。

HPL_E20_AS_CAP2 / DURA-LU：未同时满足数值门槛。
对 F20：clean -0.306349 pp；六景 -0.790575 pp。
对 F21：clean -0.478373 pp；六景 -1.010317 pp。

逐种子准确率与 Macro-F1 见 target_per_seed.csv；逐 RX／TX／day／RX×day 见 target_strata.csv；DURA 相对同一模型 raw 的纠错、损伤与净纠错见 target_paired_raw.csv。全部原定种子和场景均保留。

## 方法、预算与数据口径

源固定设计28431a31的完整NativeAs梯度帽对照，从原同seedF21E20独立native10+20到E50/11100=13L37LU，不加载候选权重或原E30。相对固定NativeAs cap1仅将t2/cls_head/other组U/L范数帽1改2；负点积投影、tinyL1e-12、完整当前L anchor、一次消耗完整U/nativeAdamW/clip5保持。原rawargmax/.95/U权重6/六学生/nativeAs半批/LS.01/RDrop1/AsKL0/浅中层.1/joint.05/sharedstyle/CE1.25/原L/defaulthead均保持，无额外autograd/teacher/student/head前向或推理参数。新6660步，As850500/物理学生1701000/有效AsCE视图2rawacceptedhalf/rawfallback0；累计teacher2097900/studentU10602900。完整源19980既有梯度记录的固定cap1/cap2算术三seed均通过预定规则，CPU1/65.676s/no new model or IQforward；该算术不是新训练、真实clipping或目标收益，Joint来源向完整NativeAs转移须独立验证。真实原E20检查证明至少一组实际U>L且所有组U<=2L/负点积投影/完整L显式梯度，之后丢弃探针重载原父。实际U准确率UNKNOWN，无Utruth/target/G训练；非负投影不保证AdamW或目标单调。固定NativeAs cap1匹配参考及F20/F21双控制raw+DURA42cell；此前目标曝光保留，不按本次目标分数调参、选父/seed/checkpoint或选择性重跑。本方法以完整原L六前向目标为anchor，U权重6，逐t2／cls_head／other组仅投影与L冲突的U方向，U范数帽为2倍L范数、tiny-L平方范数阈值1e-12，显式梯度接回原native backward/AdamW，未重复计算CE梯度；实际6660次guard及冲突／帽计数随两段累计。无额外前向，有额外autograd成本，不保证AdamW步／L损失／目标性能单调。

每景 168000 条、三个 F 模型 seed、既有 F0 未见 RX 测试物理集；source V 无状态更新。每行全部 14 项 raw／DURA 决策固定后独立 CPU truth-last；本版全部方法及三个seed齐全后统一读取结果。所有七景、逐 seed、RX／TX／day／RX×day 与修复损伤均保留，TX 编号 0 至 5 依次对应 target_summary.json 的 class_order，Macro-F1 为 0 至 1 的比例。不是新的独立盲测。

G 共用用户固定的 DURA v1.0 R69、单个 G seed2026092701、266951 参数，Graph-RR reuse=True／graph=True／native=False，FP32／TF32off。G 原训练使用合法 L/U 源 IQ，无 TX 标签且 F-free；F20 监督描述其 F，DURA＋F20 整体仍包含 G 的 U 权限。这里三个 seed 的 SD 仅跨 F，未扫描 G，不将 G 用于新 F 训练。

源 U 唯一记录 56700，累计无标签训练曝光均 2097900。普通 PL 对照三 seed 的实际二值接受量为 1987934／1992314／1989215；接受率或连续权重不是伪标签正确率。新机制的实际物理曝光、学生视图与调用次数逐 seed 报告，不宣称相同训练 FLOP。父阶段、增量和累计训练经过时间已保存；共享 GPU 墙钟时间不能视为独占 GPU 小时。evaluation_completion.json 的逐景 batch128 预测时间包含 raw 与 G＋F、输入传输和同步，不能替代 batch1 部署时延，不套用旧 F50 配对的 4.531972 ms。

## DURA 对同一 F 的净纠错

下表合并三个 seed，clean 分母 504000，六景分母 3024000。修复为 raw 错而组合对，损伤为 raw 对而组合错；六景复用相同物理记录，不视为新增独立样本。它描述 G 对同一 F 的作用，不能替代新 SSL 对两个匹配对照的收益。

| 方法 | 范围 | 修复 | 损伤 | 净纠错 | 准确率差（pp） |
|---|---|---:|---:|---:|---:|
| F20 | clean | 1012 | 648 | 364 | +0.072222 |
| F20 | six_mean | 401201 | 181698 | 219503 | +7.258697 |
| F21 | clean | 908 | 667 | 241 | +0.047817 |
| F21 | six_mean | 412101 | 178791 | 233310 | +7.715278 |
| HPL_E20_AS_CAP2 | clean | 1066 | 612 | 454 | +0.090079 |
| HPL_E20_AS_CAP2 | six_mean | 365186 | 224630 | 140556 | +4.648016 |

## 判定

HPL_E20_AS_CAP2 / raw：未同时达到原数值门槛。相对 F20，clean -0.324206 pp，六景 +1.820106 pp。相对 F21，clean -0.520635 pp，六景 +2.056944 pp。

HPL_E20_AS_CAP2 / DURA-LU：未同时达到原数值门槛。相对 F20，clean -0.306349 pp，六景 -0.790575 pp。相对 F21，clean -0.478373 pp，六景 -1.010317 pp。

完整结果不回流修改已冻结的其他候选参数、父来源或测试矩阵；当前默认部署 PAMF／DURA 版本保持其用户指定定义。
