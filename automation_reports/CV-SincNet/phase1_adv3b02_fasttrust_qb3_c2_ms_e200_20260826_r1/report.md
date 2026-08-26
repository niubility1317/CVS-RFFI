# Phase1 FastTrust-QB3 C2多seed补齐与伪标签质量审计

## 最小预登记

- run ID：`phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826_r1`
- 目标：在不改变已冻结QB3数学定义与现行`LEO_WEAK`日程的条件下，补齐seed713101、seed713102的C2 E200结果；同时建立`V_select-as-U`独立伪标签质量审计和真实共享参数梯度遥测。
- 候选矩阵：seed713101 C2、seed713102 C2；C2为H+P-set，P-conditional关闭。
- 固定协议：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，source-only，U_s训练期不读取TX真值，target/query不参与训练、校准、选模或调参。
- 训练预算：E200、U batch256、`eval_batch_size=512`、逐epoch恢复checkpoint。
- 终评：Clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分别保存。
- 预期artifact：每行200条epoch记录、`final_ssdg.pth`、四场景独立指标与日志；`V_select-as-U`truth-blind artifact及独立truth-last评分结果。
- 技术停止：协议/query越权、错误seed/split/checkpoint、输出覆盖、错误checkout、同一确定性异常至少两行、prediction无法闭合或进程归属不清。低性能不停止。
- 当前状态：`RUNNING`。
- 精确启动命令：`env ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_qb3_c2_ms_e200_a185bb7f bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_qb3_c2_ms_e200_a185bb7f/code/scripts/launch_phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826.sh`

## 需求追踪

| ID | 报告要求 | 实现面 | 验证状态 |
|---|---|---|---|
| `QB3-P0-QUALITY` | `V_select-as-U`生成truth-blind逐样本artifact，独立连接truth评分 | 独立三进程审计与评分文件 | `COMPLETED` |
| `QB3-P0-GRAD` | 对实际H/P损失与共享参数求导，报告范数比与余弦 | `code/SSDG/train_ssdg.py`及聚焦测试 | `IMPLEMENTED_RUNNING` |
| `QB3-P1-C2MS` | 仅新增seed713101、seed713102的C2 E200 | 新matrix与launcher | `RUNNING` |
| `QB3-SPEED` | 缓存冻结anchor clean logits并向量化路由预算 | 训练路径与速度A/B | `ANCHOR_CACHE_LOCAL_VERIFIED` |
| `QB3-SINC` | `torch.sinc`+FP32滤波器合成，独立匹配验证 | `code/model.py`及数值测试 | `REAL_CKPT_SMOKE_PASS` |
| `QB3-RG` | P-set/P-cond独立预算与rank风险门控 | source-only候选 | `PENDING_P0_EVIDENCE` |

## 口径冲突处理

附件将`mixed_orbit`描述为正式默认，但当前`项目.md`明确Phase1默认采用三类`LEO_WEAK`日程。本轮按当前`项目.md`执行，附件中的历史`mixed_orbit`主张不进入本次矩阵。

## 本地实现与验证

- `code/cvsrffi/phase1_pseudolabel_quality.py`：实现truth-blind逐样本记录和独立H/P质量评分，包含class、receiver、receiver/day、class×receiver分组。
- `code/scripts/phase1_fasttrust_vselect_quality.py`：`generate`只接收移除TX字段后的V_select-as-U数据；`extract-truth`和`score`为独立子命令。
- `code/SSDG/train_ssdg.py`：将H/P梯度遥测绑定到实际loss与共享参数，断图记录`NaN`而非0，并增加相对labeled梯度余弦。
- `configs/phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826.json`：仅包含seed713101、seed713102的C2 E200。
- `code/scripts/launch_phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826.sh`：专用绑定C2矩阵，拒绝误落入旧速度profile默认值。
- 独立P0/P1审查首轮发现同进程truth接触和launcher默认矩阵两项问题；定点修复后复审结论为`READY`，未增加白名单外gate。
- 本机Git Bash探针为`MSYSTEM=`，不满足`MINGW64`，本地`.sh`通道记为`FAILED`；远端发布后执行`bash -n`和behavioral dry-run。
- Windows原生`ssr-gpu`验证：聚焦测试`58 passed`；三个Python实现文件通过`py_compile`；`git diff --check`无空白错误。

## 真实checkpoint smoke修复

- 首次远端`V_select-as-U generate`在进入模型前触发`KeyError:0`。根因是无标签dataset edge已返回独立domain张量，生成器却再次把去除TX字段后的metadata传给`domain_from_extra`。
- 失败进程没有生成artifact，失败输出目录不复用；新增无标签batch解包回归测试后，生成器直接使用dataset edge返回的domain张量。
- 修复后的聚焦测试为`4 passed`，脚本通过`py_compile`；等待新Git提交与新release执行真实checkpoint复验。
- 新release复验继续推进后在receiver读取处触发`KeyError:1`：truth-hidden batch已被解包为metadata字典，但生成器仍调用只接受训练期原始batch包装的解析器。该次同样发生在推理前且没有生成artifact。
- 新增metadata字典直读回归测试，receiver改为从已解包metadata构造只含观测域信息的张量；相邻聚焦回归为`57 passed`，待再次提交并用全新release/output root复验。
- 第三次复验进入RC4路由后出现CUDA gather越界。定位为dataset edge返回的raw domain ID未按checkpoint的`domain_label_map`映射为紧凑域索引；训练主路径本来执行该映射，问题仅在新增审计脚本。
- 审计脚本现复用训练主路径的`domain_from_extra`映射并对未注册域fail-closed；增加`{3:0,4:1}`紧凑映射回归，聚焦回归仍为`57 passed`，后续使用全新release/output root验证。
- 第四次真实生成已成功产出`12,600`条truth-blind记录，生成进程明确报告`truth_access=false`。随后独立truth导出进程因有标签batch仍采用`[domain,metadata]`包装而在字段读取时报错，未生成truth sidecar，评分未开始。
- 统一字段读取器现同时支持truth-hidden字典和训练期batch包装，并增加两种形态的回归；聚焦回归为`57 passed`，将以新release继续独立truth导出和评分。

## 伪标签质量实测

- 固定checkpoint：历史C2`E200_C2_BC_H_PSET/final_ssdg.pth`；审计release为提交`a185bb7fd28ef36703e1a721399e050b09ba0425`。
- 三个独立进程依次完成truth-blind生成、truth sidecar导出和连接评分；artifact与truth均为`12,600`条，ID集合严格相等。
- 路由计数：H=`2,886`，P=`1,133`，R=`8,581`；H覆盖率`22.9048%`，H精度`99.7574%`，H-AURC=`0.0001604`。
- P-set覆盖率`99.0291%`，平均集合大小`2.098`，P95集合大小`3`，set-safe样本内top-1排序准确率`96.1676%`。
- receiver最弱H精度`99.0244%`，receiver/day最弱H精度`97.6744%`；receiver最弱P-set覆盖`98.3471%`，receiver/day最弱P-set覆盖`97.5309%`。
- class×receiver最弱H精度为`93.1034%`（class1/receiver0），最弱P-set覆盖为`50.0%`（class2/receiver5）。这表明总体质量很高，但小样本交叉单元仍是后续RG风险门控应重点处理的长尾区域。
- 完整分组结果见`c2_existing_checkpoint_quality_score.json`。

## 发布与启动证据

- Git实现提交：`a185bb7fd28ef36703e1a721399e050b09ba0425`；本地与`origin/work/cvs-active`OID一致。
- release归档：本地`E:\type10-7\release_artifacts\phase1_fasttrust_qb3_c2_ms_e200_a185bb7f.tar.gz`映射到远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_qb3_c2_ms_e200_a185bb7f.tar.gz`，唯一传输SHA256为`7984b2b2f2def2d4cc8e884fa47912521727061204823a478ce8d21a7e15f503`，远端校验通过。
- 远端三个Python实现文件编译通过，专用launcher与矩阵launcher的`bash -n`通过；behavioral dry-run只产生GPU4/seed713101与GPU5/seed713102两行，均为C2 E200、U256、eval512、逐epoch恢复checkpoint和Clean+三类LEO弱场景终评。
- 2026-08-26 13:42 CST启动前GPU4、GPU5均为空闲，run root与launcher日志路径均不存在；GPU0另有独立任务，未触碰。
- detached launcher PID=`3556728`，CWD和cmdline均绑定上述release与run ID；训练PID=`3556760`映射GPU4/seed713101，PID=`3556764`映射GPU5/seed713102。
- 两行均已写入epoch1，训练日志从`6,911`字节增长到`12,431`字节，metrics文件已落盘；GPU4/GPU5显存约`3.7/3.6GB`，无`Traceback/CUDA error/RuntimeError/Exception`指纹。当前最高可证状态为`RUNNING`，尚无最终性能结论。

## Sinc数值修复

- 原实现用`sin`差除以`πt`再覆盖中心点；本地FP16定点测试复现`low_hz_`梯度NaN，验证了报告中的技术风险，而不只是静态推断。
- 新实现使用解析等价的`2f₂·sinc(2f₂t)-2f₁·sinc(2f₁t)`，滤波器参数、时间轴、窗函数、滤波器合成和卷积全部在autocast外以FP32执行，输出再恢复输入dtype。
- 第一轮没有改变`abs+clamp`参数化，没有启用`torch.compile`，因此改动只隔离数值稳定性变量。
- FP16、BF16、FP32、极端low/high frequency、forward有限、参数梯度有限、初始scale=`65,536`的CUDA GradScaler以及连续1,000优化步全部通过；与旧FP32滤波器匹配。Sinc及相邻模型测试共`28 passed`。
- 独立P0/P1审查结论为`READY`，未发现会导致真实smoke跑错、AMP/dtype语义错误或改变滤波器数学定义的问题。该修复尚未进入当前运行中的C2 release，将通过独立release做短smoke和同seed匹配验证。
- Sinc独立提交为`eb0db1dd5085ef6a5f0e67e996525905d22cd4b8`；release归档SHA256为`d502bff4d6096a08a464c0c92df340f8070a4dcb42b263729546b19983e34086`，远端传输与编译通过。
- N607训练环境没有pytest模块，远端pytest入口记为`FAILED`且未安装任何包；改用同release原生Python在GPU7验证FP16输出、初始scale=`65,536`、loss=`0.0004073`及两组Sinc参数梯度全部有限。
- 同release进一步加载真实历史C2 checkpoint，对truth-hidden V_select执行完整前向并生成`12,600`条记录，`truth_access=false`。因此当前证据为`REAL_CKPT_SMOKE_PASS`；尚未进行同seed C0/C3短程匹配训练，不宣称性能等价。

## 冻结anchor缓存速度优化

- 新增显式开关`rc4_cache_anchor_logits`，默认关闭；仅在`fasttrust_rc4=true`、`rc4_use_anchor=true`且`rc4_lambda_feature_anchor=0`时允许启用，避免只缓存logits却误服务需要anchor feature的损失。
- 训练前只遍历U_s的确定性clean view，以稳定`base_index`建立保持anchor实际AMP dtype的GPU驻留dense logits表；不读取TX真值，不接触target/query，缺失、重复、越界ID均fail-closed。
- 缓存预计算前后完整保存并恢复Python、NumPy、CPU与CUDA RNG状态，不改变正式训练随机序列；缓存只存在于训练进程内，不写入checkpoint、release结果或deployment bundle。
- live与cached anchor前向使用完全相同的AMP开关。独立审查首轮发现两者AMP上下文不一致的P1，修复后定点复审结论`CLOSED`。
- 本地缓存、RNG、launcher和既有QB3相邻回归共`30 passed`，Python编译通过；预登记E6同seed配对矩阵只改变`cache_anchor_logits=false/true`，GPU6/7、seed392002、U256、eval512和逐epoch恢复点完全一致。
- E6的主要速度判据为epoch2–6的`muse/time_train_batches_s`与`muse/u_samples_per_s`配对差异；另报告一次性`[RC4-ANCHOR-CACHE] build_s`并给出6epoch及E200摊销，避免只看epoch内速度而忽略缓存构建成本。
- 首次速度launcher尝试`phase1_adv3b02_fasttrust_qb3_anchor_cache_speed_e6_20260826_r1`在训练进程创建前即退出：专用launcher默认把`CODE_ROOT`指向项目根，而归档只解压到独立release。该尝试没有run root、checkpoint或训练PID，仅保留启动失败日志，状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 修复后专用launcher默认从自身路径解析release根；为保持run ID和输出不可覆盖，真实速度验证改用新ID`phase1_adv3b02_fasttrust_qb3_anchor_cache_speed_e6_20260826_r2`，矩阵科学变量不变。
- `r2`两行均完成E6及Clean+三类LEO弱场景终评。epoch2–6训练batch均值由live的66.102秒降至cached的62.258秒，下降5.815%；U吞吐由720.05升至756.76样本/秒，提高5.098%。计入5.882秒缓存构建后，E6训练阶段净耗时下降3.997%，按稳定epoch外推E200约节省762.9秒（12.7分钟）。
- 但`r2`不能晋级：缓存实现把AMP anchor logits强制提升为FP32，而live路径保持FP16，导致路由与训练轨迹不等价。完整逐epoch比对发现3,577个非时间有限数值中383个不同；epoch4验证正确数相差78。终评也出现差异：live/cached的Clean为87.2217%/87.3383%，LEO均值为68.4306%/67.9483%，receiver×LEO floor为49.5583%/49.4167%。因此`r2`结论限定为`SPEED_FEASIBLE / SEMANTIC_EQUIVALENCE_FAILED / NO_PROMOTION`。
- 已将dense cache改为保留真实anchor输出dtype，不再`.float()`；新TDD覆盖FP16 cache与lookup dtype。修复验证改用全新run ID`phase1_adv3b02_fasttrust_qb3_anchor_cache_speed_e6_20260826_r3`，仍只比较cache开关。
- `r3`使用修复后的FP16 cache完成E6及四场景终评。epoch2–6训练batch均值由66.309秒降至64.349秒，下降2.956%；U吞吐由717.32升至737.91样本/秒，提高2.870%。计入6.061秒构建后，E6训练阶段净下降2.374%；按稳定epoch外推E200节省386.0秒（6.4分钟）。
- `r3`仍存在跨进程非确定性：live/cached的Clean为86.7983%/85.8950%，LEO均值为69.4128%/68.9906%；同时相同live配置在`r2`与`r3`间Clean自身波动0.4233个百分点。因此E6准确率差异不归因为cache，不作为方法晋级或退级证据；cache只按数学路径、dtype和速度证据判断。
- 为消除GPU6/7固定速度差，新增`r4`交叉位置复验：live移至GPU7，cached移至GPU6，其余配置不变。最终速度结论以`r3+r4`交叉均值为准。
- `r4`已完成训练、Clean及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`终评，无Traceback、RuntimeError或OOM。GPU角色交换后，epoch2–6训练batch下降8.831%，U吞吐提高8.875%；计入6.530秒构建后，E6训练阶段净耗时由435.603秒降至405.684秒。
- `r3+r4`交叉均值为：稳定epoch训练batch由69.153秒降至64.994秒，下降6.014%（1.0640×）；U吞吐由690.49升至730.24样本/秒，提高5.756%；计入平均6.296秒构建后，E6净耗时下降4.708%。按E200稳定epoch线性外推约节省825.5秒，即13.76分钟。完整数值见`anchor_cache_speed_cross_results.json`。
- 速度优化结论：缓存机制通过实现、TDD、独立P0/P1复审、两个GPU位置交叉E6及四场景artifact闭合，允许作为后续正式release的训练速度优化；开关仍默认关闭，当前已运行的正式C2不热补丁。E6准确率波动不用于cache性能归因，也不替代E200方法验证。

## 正式C2训练、artifact与分析闭合

2026-08-27 00:08 CST完成最终只读核验。两个训练PID均已正常退出，GPU4、GPU5已释放；dispatcher和两行日志均未出现Traceback、RuntimeError、CUDA OOM、Killed或确定性错误指纹。本节严格区分三个状态：

- 训练完成：`VERIFIED`。seed713101、seed713102均具有连续E1–E200的200条JSONL记录和200条CSV记录，训练完成receipt为true，final checkpoint epoch均为200。
- artifact完整：`VERIFIED`。两行均具有`final_ssdg.pth`、Clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`独立JSON与日志，以及15个receiver×LEO单元的联合指标。每个场景评测60000条样本、5个未见receiver、每个receiver12000条；严格加载开启，missing/unexpected/shape mismatch均为0，未使用fallback。
- 分析完成：`VERIFIED`。已下载并逐行解析两行全部结构化epoch记录，全文扫描训练、四场景和联合评测日志及dispatcher日志；C0/C3沿用此前同样全文解析的同row摘要，不用tail或样本替代最终分析。

两行final checkpoint大小均为15758996字节；seed713101、seed713102墙钟分别为9.738小时和9.631小时，峰值显存均约3.205GiB。本run最高交付状态为`ANALYZED`，但交付闭合不自动等于科学晋级。

## 正式C2逐seed结果

| seed | Clean | Clean receiver floor | leo_clear_weak | leo_low_elev_weak | leo_rain_weak | LEO均值 | 场景floor | receiver×LEO floor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 392002 | 84.8867 | 70.5583 | 75.6667 | 73.0567 | 72.4333 | 73.7189 | 72.4333 | 57.0750 |
| 713101 | 85.5433 | 74.4917 | 75.5267 | 72.9883 | 73.0333 | 73.8494 | 72.9883 | 58.5583 |
| 713102 | 84.1333 | 70.2917 | 74.7617 | 72.7067 | 72.4667 | 73.3117 | 72.4667 | 57.4417 |
| 三seed均值±样本标准差 | 84.8544±0.7056 | — | — | — | — | 73.6267±0.2805 | 72.6294±0.3113 | 57.6917±0.7726 |

这里没有拼接不同seed或不同候选的单项最大值；每一行都对应同一个seed、同一个候选和同一个final checkpoint。

## C0↔C2↔C3同row因果拆分

C0关闭U身份损失，C2增加H+P-set，C3在C2上增加P-conditional。因此在冻结的同seed比较中，`C2-C0`解释H+P-set联合贡献，`C3-C2`解释P-conditional增量，`C3-C0`只解释二者合计；不能把C2-C0全部写成P-set单独贡献，因为H也同时开启。

| seed | 候选 | Clean | LEO均值 | 场景floor | receiver×LEO floor |
|---:|---|---:|---:|---:|---:|
| 392002 | C0 | 84.4000 | 73.5589 | 72.2083 | 57.2417 |
| 392002 | C2 | 84.8867 | 73.7189 | 72.4333 | 57.0750 |
| 392002 | C3 | 85.2017 | 73.8667 | 72.5450 | 57.6167 |
| 713101 | C0 | 84.7050 | 73.6006 | 72.7333 | 58.4000 |
| 713101 | C2 | 85.5433 | 73.8494 | 72.9883 | 58.5583 |
| 713101 | C3 | 85.7067 | 73.8411 | 72.9267 | 58.2750 |
| 713102 | C0 | 84.0067 | 73.1267 | 72.2100 | 57.7833 |
| 713102 | C2 | 84.1333 | 73.3117 | 72.4667 | 57.4417 |
| 713102 | C3 | 84.6383 | 73.5389 | 72.6717 | 58.0250 |

三seed配对平均贡献如下：

| 因果增量 | Clean | LEO均值 | 场景floor | receiver×LEO floor | 一致性 |
|---|---:|---:|---:|---:|---|
| H+P-set：C2-C0 | +0.4839pp | +0.1980pp | +0.2456pp | -0.1167pp | 前三项3/3为正；receiver×LEO仅1/3为正 |
| P-conditional：C3-C2 | +0.3278pp | +0.1222pp | +0.0850pp | +0.2806pp | seed713101的后三项为负；其余2seed为正 |
| 合计：C3-C0 | +0.8117pp | +0.3202pp | +0.3306pp | +0.1639pp | Clean和LEO均值3/3为正；receiver×LEO为2/3为正 |

三seed候选均值为：C0的LEO均值/场景floor/receiver×LEO floor=`73.4287/72.3839/57.8083`，C2=`73.6267/72.6294/57.6917`，C3=`73.7489/72.7144/57.9722`。所以C2稳定提高平均LEO和场景floor，但没有保住最坏receiver×LEO单元；P-conditional在均值上补回局部floor，却不是逐seed稳定修复。

## 伪标签质量与真实梯度利用结论

独立truth-last质量审计已经证明：H覆盖22.9048%、精度99.7574%，最弱receiver/day H精度97.6744%；P-set覆盖99.0291%，平均集合大小2.098，set-safe条件排序准确率96.1676%。因此当前主要问题不是总体伪标签精度不足，而是交叉长尾和单位伪标签的有效梯度利用不足。class×receiver最弱P-set覆盖只有50.0%，与C2的receiver×LEO floor没有稳定改善相互一致。

修复后的真实共享参数遥测在两行的E1/E41/E91/E161/E181/E200均触发。E41以后H与P-set梯度不再错误记录为0，证明计算图绑定已修复；但`g_identity/g_L`多处只有约`1e-8`至`2.5e-5`，H/P-set相对labeled梯度余弦在正负之间波动，没有形成稳定同向信息。结论是“高质量伪标签已经进入实际梯度图，但利用强度偏弱且方向不稳定”，不能声称当前预算已充分转化为表征收益。

当前release仍复现旧Sinc首batch异常：两个seed均在E1 batch1首先出现`id_backbone.sinc.low_hz_`的24个NaN梯度，loss本身有限；两行估计分别跳过36和38个非有限梯度batch，约占0.0870%和0.0918%。该异常不是C2特有，训练和终评仍完整闭合；已经完成的`torch.sinc`+FP32滤波器合成修复将在下一release使用，不对本run热补丁或重写结果。

## 训练速度最终结论

当前C2正式行没有启用后续anchor cache，因此9.63–9.74小时墙钟不能用于宣称缓存后的E200实测加速。独立r3+r4交叉E6已给出稳定epoch训练batch下降6.014%、U吞吐提高5.756%、计入构建后E6净下降4.708%，线性外推E200节省约13.76分钟；这一结论只证明工程速度优化可进入下一正式release，不替代下一候选的E200速度读回。

下一release同时启用已验证的FP16语义保持anchor cache和Sinc数值修复；两者分别解决重复anchor前向和共享前端NaN，不改变伪标签科学变量。正式结果仍需报告真实E200墙钟，不能只复述短跑外推。

## 晋级判定与下一候选

- 实验交付：`ANALYZED`。
- C2科学信号：`MULTI_SEED_POSITIVE_MEAN_AND_SCENARIO_SIGNAL`。
- C2默认方法晋级：`NO_PROMOTION_TO_DEFAULT`。
- C3/P-conditional晋级：`NO_PROMOTION`。

不晋级的直接原因不是低总体质量，而是C2的receiver×LEO floor相对C0在2/3seed下降、三seed均值下降0.1167pp；P-conditional虽然平均补回0.2806pp，但seed713101反向，不能当作稳定floor修复。同时，class×receiver P-set覆盖最弱单元只有50%，真实身份梯度强度仍极小且方向波动。

下一候选冻结为`C2-RG1`：保留H+P-set、继续关闭P-conditional，不扩大H/P预算和唯一伪标签数量；只根据source-only质量审计加入receiver×class小样本风险门控，使证据不足的P样本退回R，并对实际共享参数做轻量梯度归一化/上限约束，目标是提高单位U身份信息而不是增加覆盖。工程侧启用FP16 anchor cache和Sinc FP32修复。首轮仍采用单seed同row C0↔C2-RG1最小矩阵；晋级门要求LEO均值与场景floor不低于当前C2，同时receiver×LEO floor相对同seed C0不下降。不得使用target结果调receiver、class或阈值参数。

## 可复算交付物

- 完整分析脚本：`analyze_c2_results.py`。
- 机器可读终局摘要：`c2_final_analysis_summary.json`。
- 伪标签质量摘要：`c2_existing_checkpoint_quality_score.json`。
- anchor cache交叉速度摘要：`anchor_cache_speed_cross_results.json`。
- C2两个正式行的完整本地artifact副本保存在非Git分析输入目录`detailed_analysis_inputs/`；Git只发布本次报告、分析脚本和机器可读结果，不推送checkpoint或大体积训练日志。
