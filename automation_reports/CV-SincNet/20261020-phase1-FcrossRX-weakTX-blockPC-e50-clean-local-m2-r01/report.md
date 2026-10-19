# F＋跨 RX 同 TX：弱类模块梯度协调

状态：LOCAL_VERIFIED，准备在本机发布首 seed 两条件。

基底为原始 F CROSS_RX50（三路 SIG_TIME、统一居中物理 Sinc、RDrop1、旧门控＋ownEMA＋cos、普通增强／LS.01、默认头正常联合）。历史 85.1139% 只作方法锚点；增量复用本机原始匹配 BASE 三 seed，不重复训练或测试。无星地信道增强，测试只跑 clean。

| 条件 | 实际特征梯度 | 首种子 | 源域通过后的原样扩展 |
|---|---|---|---|
| NORM_L_ID | 保持 L 身份目标方向，逐模块匹配协调后的该分量范数 | 2026092701 | 2026092702／03 |
| BLOCK_PAIR | 两个弱 TX 类 CE 梯度在模块内冲突时去除相反分量 | 2026092701 | 2026092702／03 |

每模型从零 50 轮＝33L＋17U、11100 次主优化器更新。前 5 轮不变，随后每 4 步计算：第 1112 至 11100 步，共 2498 次计划事件；强度从第 6 轮 0.1 增至第 15 轮 1。合格事件额外 3 个特征 VJP，最多 7494／模型。新增模型前向、IQ 读取、参数、真实优化器更新、头拟合均为 0。

弱类固定为源 V 已确定的 14-7／20-19（索引 1／3）。类 CE 是原有两次随机 RDrop 对同一次增强 L128 的每样本 CE 均值，该类求和／128，不增加类权重。gL 为当前 L 身份 RDrop＋原浅／中层 CROSS_RX SupCon 梯度，不包含 U 或其他原生目标。

对每模块 ga·gb<0 时，同时令 a′=a−(a·b)b/||b||²，b′=b−(a·b)a/||a||²，delta=η(a′＋b′−a−b)。BLOCK_PAIR 使用 gL＋delta；NORM_L_ID 保留 gL 方向并匹配 ||gL＋delta||。只匹配同一当前状态下的 L 身份分量，不能声称两条轨迹或完整组合梯度范数相同，也不保证实际 AdamW 下降。无冲突、零主梯度、零弱类梯度或缺弱类时不修正。

零值线性代理 ∑(p−detach(p))·detach(delta) 将修正送入实际部署的全部 id_backbone 特征，包括三路 SIG_TIME 与响应分支，排除默认 cls_head。原损失数值不变；未使用参数保持 None，避免意外开启 AdamW 衰减。默认头正常联合，新增项直接 W 梯度为 None；后续全局裁剪可能改变主头梯度比例。

Sinc 模块仅低频与带宽两个 q 叶参数，共同尺度 25000 Hz；模块投影系数及范数比对共同单位缩放不变。原 p=p0+25000q、Fs=25 MHz、物理 Hz clip5、坐标 lr／eps 和包含 origin 的物理衰减保持，不叠乘历史倍率。

## 权限、扩种子与自动测试

复用既有 READY／EXACT_MATCH 源角色与 VALIDATED_ONCE clean capsule，不重建数据。源 RX1／3／4／6／8、day1／2／3，split_seed392005；L6300／U56700（TX 隐藏）／V27000。新训练从零，仅自身 EMA；匹配 BASE checkpoint 只用于一次无 query smoke 并丢弃。

源 V 固定 no_grad 默认头健康与 actual joint160 描述性几何，不拟合分类器或原型决策；out_RX 仅排除描述质心中的 RX，所有源 RX 均已参与训练，不称为未见 RX 模型验证。

首 seed 两条件源训练完成后，先按固定规则独立决定扩展，再访问新目标汇总：每个弱 TX out_RX marginq10≥同 seed BASE＋0.002，RXspread／gap≤1.05 倍，零特征 0；默认头整体≥−0.2 个百分点、最差 RX≥−0.5 个百分点、任一 TX≥−1 个百分点；2498 次计划、合格≥99%、非零修正≥10%。只通过条件原样扩展 02／03，失败条件仍自动完成首 seed clean 测试，不按目标结果改参数、补 seed 或重跑。

每行固定 final50 student 和 checkpoint 内默认头，clean168000 全预测完成后独立 CPU truth-last 评分。报告逐 seed 配对、实际 n、均值、样本 SD、正差数、6 TX 召回、Macro-F1、最差 RX／day、U 曝光与接受率和成本。源证据不能替代目标提升结论，全部负结果保留。

## 发布与验证

环境：native Windows Python3.10／PyTorch2.10+cu128、FULL_FP32_NO_TF32、threads2／workers0。CWD：IMMUTABLE_LOCAL_RELEASE/cvs_F_weakTX_blockPC50_local_20261020_r01。输出：LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-blockPC-e50-clean-local-m2-r01，不可覆盖。物理 GPU0 全 owner 最多 3 个 source／predict worker，派发前 RAM≥4.5 GiB；独立 CPU scorer 最多 1 个。保留现有健康任务。

启动命令：

LOCAL_BASE_PYTHON -X utf8 -u -m experiments.cvs_f_weaktx_blockpc50.driver --profile LOCAL_PROFILE --metadata LOCAL_SOURCE_NATIVE_GATE_METADATA_R02 --base LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-blockPC-e50-clean-local-m2-r01 --reference-base LOCAL_RUN/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01 --commit RELEASE_COMMIT --max-workers 3

代码身份：本次实际 code／config Git commit；发布后写入 launch_verification.json 和每行 source_commit。一个发布包复制核对、提取后编译和本机 preflight，不增加成员哈希／签名链。停止规则：Role/provenance/config/output conflicts, nonfinite loss/gradient, incomplete native budget/EMA/Sinc/feature scope or failed worker stop new dispatch; preserve outputs and other healthy workers; no automatic or performance-based retries

预期产物：初始化／原生参数、完整50轮／11100步日志、Sinc／RDrop／SupCon 观察、weak_pair_progress／steps／training_observation、source_frozen、source_expansion／frozen_matrix、clean prediction、独立 score、配对汇总。原始 IQ、weights、logs、逐包预测与本机进程资料留本地；Git 仅提交实现、配置和精简结果。

聚焦检查 PASS：独立投影数值、类贡献归约、Sinc 共同单位尺度、无冲突／零／相反向量、未使用叶参数 None、配置负例、真实 F 梯度 oracle、默认 W 辅助 None／主梯度精确保持、原两次前向／损失值／CPU RNG、EMA／export160、900722参数、一次匹配 real-checkpoint no-query smoke。CUDA 两条件联合反传和 CPU／CUDA RNG 保持 PASS。检查无源／目标 IQ 读取和真实更新，非性能结果。一次独立 P0／P1 审查 PASS，无直接阻断项。

此前原三 seed BASE 源 L 全量梯度诊断发现两个弱类池化 CE 特征梯度夹角均为负；只是冻结 clean／eval 证据，不代表训练每批冲突或实际 AdamW／目标效果。本路线受 [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) 启发，是固定弱类、分模块、有范数对照的适配，不宣称完整复现、首创或预设有效。

REJECTED_EXTRA_GATE：通用设计审批暂停、全项目测试不属白名单；用户已有有界研究授权，采用聚焦验证及一次独立 P0／P1 审查。已暴露 Phase1 clean 探索，不声称独立盲测、Phase2 或星地部署。
