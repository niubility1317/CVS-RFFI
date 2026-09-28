# D92-SGJoint交付与验收边界

## 用户确定的范围

固定现有Phase1，改进各K的新旧类综合H，同时要求新类表现改善并限制旧类退化。辅助训练只使用当前row的合法卫星support；不使用或传输source IQ、源域逐样本特征，不重新访问source loader，不训练新的源域辅助模型。

用户在了解数据复用与独立验证的区别后，已明确改为“先复用数据”。因此当前安排为两个预登记的重复基准实验，复用已验证received IQ、相同support/query、固定Phase1特征、冻结地面摘要及原D92基线预测。SGJoint公式与配置保持不变，只用各row合法support拟合和选参；不以query评分调参。结果明确标注`REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS`，不宣称新独立确认。

## 交付内容

算法说明见[D92_SUMMARY_JOINT_DESIGN_20260928.md](D92_SUMMARY_JOINT_DESIGN_20260928.md)。核心实现为`code/cvsrffi/stage2_d92_summary_joint.py`，预测入口为`tools/predict_d92_summary_joint.py`，固定配置为`configs/d92_summary_joint_frozen_20260928.json`。

- 既有合法量化地面摘要只决定冻结的类无关identity归一化算子，不扩充support，不训练地面类条件头，不生成伪源数据。
- 同一received IQ产生identity160与FFT96，所有注册类使用相同的均值、共享收缩协方差和均匀类别先验。
- K≥2仅在当前row真实物理support中交叉验证。训练fold与留出fold物理ID互斥，候选选择兼顾旧新组风险。K1用事前固定公式，不伪造holdout或借用其他row。
- query只做逐样本、全注册类推理。预测先固定，再由独立scorer连接truth。程序没有query truth训练接口。
- 完整候选/fold审计保存JSONL，紧凑逐row记录同时保存JSONL与CSV。解析解没有epoch、学习率或梯度，这些字段为null并解释原因。

该候选由未接触既有目标评分的独立设计任务推导。已有SFHead与本候选的类别先验差别来自目标函数检查；不得将该数学差别解释成已经证实的性能改善。

## 传输与本地资源

现有四个模型对应的合法量化摘要数值数组均为4412字节。NPZ与manifest两文件实际合计分别为8383、8262、8191、8361字节。读取证据保存在SFHead既有run的`evidence/ground_payload_readback.json`。这是摘要包大小，模型传输单列，不能把它当作完整部署包大小。

本候选无需新增地面统计。如果摘要已经位于接收端，则增量传输为0；否则传输上述已有摘要两文件，入口按实际文件重新计数。默认配置保守设置`summary_already_deployed=false`，只有实际部署状态能够证明已存在时才显式设为true；这不改变算法。

160×160的float64算子在卫星本地最多占204800字节，属于本地派生状态，不是星地传输量。最终LDA头按实际数组nbytes记录。Linux入口另记录整个预测进程实测峰值RSS，其范围包括输入缓存、线性代数及日志，不能宣称是卫星实测或仅分类头峰值。编码器、FFT、support拟合与query推理成本须分别说明。

## 当前重复基准与未来独立验收

当前两个run为`20260928-phase2-d92-sgjoint-repeat-rx3-m4-r01`与`20260928-phase2-d92-sgjoint-repeat-rx1-m4-r01`。前者复用RX19-1/8-14/8-7对应900splits/model，后者复用RX20-19对应300splits/model；四个模型合计4800个配对单元，包含两方法与DG共9648条评分记录。主验收按四RX的全部匹配单元等权汇总，保持每K的H/新类提升及旧类退化≤1个百分点；完整保留每cohort、RX、scene、seed及新类数结果。两个cohort都完整执行，各自全矩阵预测固定后才独立评分；两者终态前不读取新scores以改变任一后续运行。

既有输入的只读预检已核实两个原run终态、八个模型row与四个checkpoint SHA、ground/cached features绑定及两capsule的VALIDATED_ONCE。新方法只写新run路径，CPU执行，不重新读取源样本、重建数据、加载训练checkpoint或提取编码器特征。先前真实checkpoint无query smoke及缓存来源证据保持适用；本次不重新加载checkpoint，新增数据验收不成为本轮重复基准启动条件。

方法公式和选择规则在新增数据访问前冻结。先核对新增物理ID、source/target接收机互斥、checkpoint继承和合法摘要来源，再按既有最小流程登记新run；不重跑本次已完成的SCV/SFHead实验。

保持拟验收矩阵：四个冻结Phase1模型、三个LEO弱场景、K=1/5/10/20、新类数0/2/5/10/20、五个support seed。新增接收机和具体物理ID必须由新增数据证据确定，不能从当前目录日期或历史评分挑选。若沿用每场景support pool30/query30的完整26类数据口径，每个接收机每类至少需180个互不重复物理记录；三场景和support/query的物理ID互斥。数据不足时如实保留缺项，不能静默缩小矩阵或重叠物理样本。

每K在新类数2/5/10/20的相同配对单元上，要求ΔH>0、Δnew>0、Δold≥−1个百分点；同时报告全量K×新类规模、旧类单独任务、逐类floor、Macro-F1、各模型seed和各场景。所有矩阵预测完成后统一独立评分，不按中间得分筛选seed、删去困难行或选择性重跑。该门槛是当前预定验收规则，尚未证明通过。

## 本地验证状态

本地相关验证全部通过，共107个测试用例，均采用合成数据或纯控制面fixture：核心8项、预测入口15项、runner/scorer/summarizer84项。后者包括原78项接入测试和新增6项精确并列裁决测试。所有命令使用已核实的`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 -m pytest`，对应文件为`tests/test_d92_summary_joint.py`、`tests/test_predict_d92_summary_joint.py`、`tests/test_run_d92_confirmation.py`、`tests/test_d92_confirmation_score.py`、`tests/test_summarize_d92_confirmation.py`。

核心测试覆盖K1/2/5/10/20、26类、old-only、跨旧新类别重排和support重排、query分块/顺序、LDA与直接马氏距离一致、地面摘要类/域重排、不可变状态、JSON与nbytes、非法输入拒绝。入口额外实证改变或追加query不改变拟合状态或其他query分数；K2记录真实16个候选和2个物理fold；实际摘要文件字节与已部署增量0均有断言。

方法的一次独立P0/P1审查已收束，无剩余阻断项。审查与合成测试发现的两项直接问题已修复：零几何舍入噪声放大，以及SGJoint预测器和scorer的精确并列规则不一致。scorer只为SGJoint使用预登记的物理类别ID裁决，其他历史方法保留原规则。动态发布清单包含SGJoint入口及依赖。后续复用入口的新增行为另按实际改动聚焦验证，方法公式未改变；运行状态以对应run记录为准，测试通过不构成目标域性能提升证据。
