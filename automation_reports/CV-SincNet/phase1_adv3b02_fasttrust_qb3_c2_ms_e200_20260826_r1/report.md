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
