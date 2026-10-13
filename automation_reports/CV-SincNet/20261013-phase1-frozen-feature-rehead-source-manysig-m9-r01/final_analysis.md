# joint160类别无关边界与源域换头完整结果

| 特征提取器 | 原头 | 全L新cosine头 | 全L原型 | 跨RX头拟合5-shot | 跨RX头拟合10-shot |
|---|---:|---:|---:|---:|---:|
| LEGACY50 | 97.8864% | 98.0580% | 96.3951% | 95.4840% | 95.9786% |
| LOCAL_FREQ | 97.9506% | 98.1111% | 96.2605% | 95.5387% | 96.1313% |
| GLOBAL_FREQ | 97.9235% | 98.0840% | 96.1062% | 95.3584% | 95.8905% |

9个终轮模型全部完成固定诊断，297项读出、279个新普通cosine头、9900次仅W更新完整核验。全L头20轮×25批=500更新/模型；30个少样本头每个20轮×1批=20更新/模型。原Phase1骨干仍50轮，诊断新增featureupdates0，源V不参与拟合或参数选择。

导出复制ReferenceCVS的实际joint160路径，删除旧6×160 W，保留base_norm/pa_norm/gain、response_projection/response_gain和NoExplicitCFO。特征参数176065/197313/197313。所有9模型全部6300个L和27000个V的特征与原模型实际joint逐元素完全一致（最大差0），冻结参数和buffer完全不变，无旧classW。不是以h349替代。

新头普通cosine W6×160，scale30/无bias/eps1e-4，Adam LR.01/WD0，20轮，统一seeded Xavier。全L样本拟合与V评分；跨RX诊断5个源RX轮换，仅在头拟合时排除被评分RX，从另4RX的L逐类选K5/10样本，support seeds2026101301/02/03，各K5为K10子集。全部6类同公式竞争。骨干之前已经见过所有源RX/TX，此处不是未见RX骨干泛化或真实新类。

所有source_role_ID仍为原split392005的L6300/U56700/V27000，U从未打开，target从未访问；V未更新任何状态。31个新头/模型的20轮完整loss序列全部有限，训练更新预算精确；支持物理ID唯一；297项混淆矩阵准确率/Macro-F1独立复算一致。完整head-fit物理IDs、每RX/day/class-pair记录留在N607 row completion，未提交源样本、embedding、权重或原始日志。

跨模型比较：LOCAL_FREQ−LEGACY50全L新头+0.0531个百分点，GLOBAL_FREQ为+0.0259个百分点。源域的细微差异不能解释或取代原固定目标clean负结果，也不足以证明目标域特征能力增强；不据此调attention超参或重跑。

统计：表格先在每modelseed内平均固定5RX×3support draws，再对三个modelseed取均值；未把45个episode当作45个独立训练seed。全L原头/新头/原型直接每modelseed V27000统计。模型seed2026092701/02/03。完整seed结果和support波动位于analysis/row产物，均全部保留。

发布代码90e6a91f022ca740907413163097078d5512df5f，N607普通账号、source-transfer-probe独立root；实际9checkpoint无query smoke、compile、archive一次SHA比较PASS；dispatcher1579889及完整9row产物独立读回VERIFIED。每GPU总实验进程含其他owner≤3。实验用时每row4.402至5.065秒，未新增任何骨干训练。

证据：runs/20261013-phase1-frozen-feature-rehead-source-manysig-m9-r01/{preflight,launch,completion,analysis}.json及<row>/{resolved_config,source_contract,completion}.json；本地local_artifacts/source_transfer_probe_delivery/{readback,full_audit}.json。训练父来源runs/20261013-phase1-frequency-context-legacycos-clean-manysig-m9-r01/<row>/source/final_ssdg.pth，来源完整scratch，fixed final50，无基于测试的来源选择。

后续：临时分类器重建辅助训练以原骨干和历史旧门控/EMA/cos为主；真正新TX/Phase2效果须另立留TX（L/U/teacher均排除）与合法LEO support-only旧新统一测试。当前84%目标仍未达成，不能将源98%写成目标成绩。
