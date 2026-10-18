# P60：完整As适应期停止普通PA波形注入

状态：VERIFIED_RUNNING，GPU0／1 两条新尾段已实际启动；无完整源／正式性能结果。P59已完整结束并按源规则关闭，不延E50。

唯一假设：从原E20父模型继续E21–30时，普通L的PA波形扰动可能与完整practical训练形成不利交互；停止继续注入该波形、保留其他原配方，可能改善clean—urban权衡。实际PA开启不证明标签破坏，本项不预先宣称有效。

旧F＋CROSS_RX PA-bypass为留出RX1、L5040/U45360、scratchE20、无As的匹配6L，源晋级FAIL保留；ordinary-noPA为不同两前向底座，同样留出RX1/无As。本轮关键新假设是全源完整As适应期的交互，不要求先证明独立DG收益，不把旧方法改名当创新。

## 最小匹配矩阵

| 条件 | 来源 | E21–30普通PA波形 | As |
|---|---|---|---|
| 原No-As30 | 复用P43 B00原final30 | 开 | 关 |
| 原As30 | 复用P43 B01原final30 | 开 | 开 |
| NOAS_PAOFF | 原B00 own20 student＋ownEMA →新10 | 关，保留影子抽样 | 关 |
| AS_PAOFF | 原B01 own20 student＋ownEMA →新10 | 关，保留影子抽样 | 开，原三场景各1/3 |

每条新行2220更新，累计30轮/6660；两个父E1–20已经使用PA，只研究末10轮的增量，不称全程无PA。主阶段比较AS_PAOFF−NOAS_PAOFF，同时报告各新行相对原配方和交互项(AS_PAOFF−NOAS_PAOFF)−(As30−NoAs30)。原两条对照不重训、不重预测、不重评分。

新行保持原6L：128ordinary×2、128mixed×2、128raw×2；L loss总尺度1.25、SupCon .1/ORD128、RDrop同IQ、旧U/ownEMA/默认cosine头联合训练不变。无新Ag/loss/门控/分支/头。AS行新10生成142080个fullAs；NOAS行0；各行L网络前向1704960。

## 可归因实现与数据边界

实际P43链的ordinary使用Python/Torch当前设备全局RNG，没有安装其他目录中存在的PrivateAugment。不能简单no_pa=true/p_pa=0少消耗随机数。保留aug_enable_pa_normal=true、原PA概率与E13冻结日程，原simulate_pa在clone上完整抽样计算后，返回PA前prefix与零实际注入强度。只包裹新augmentor实例并恢复scope；记录shadowcalls/选中包/本会改变包数/实际注入0，不能报告PA调用0。

对同一合成输入与Python/NumPy/Torch初态，验证新返回值等于原非PA前缀，完整调用后的随机后态相同；前置操作只执行一次、PA原本位于末尾。不新增PrivateAugment、不重抽RDrop IQ。训练后模型/EMA/原U接收数可以自然变化，不强制预测逐值一致。

As64仍从raw经原normalize_input生成，不先叠普通PA/噪声/多径；旁路不扩散到As/raw/U。U真值隐藏，全部有标签As参与CE。保留几何/遮挡/传播/噪声/接收质量/补偿残差依赖。EQ=false即G=1，不偷换为弱残余。

L6300/隐藏U56700/V27000，源RX1/3/4/6/8、day1/2/3、6TX、split392005；无源RX留出/L_dev。复用未变VALIDATED_ONCE数据与固定正式IQ。分别按原P43配置核对两个scratch20父来源、EXACT_MATCH角色、noancestors/no target/final_only/ownEMA再严格加载；不加载P59、E30或E50训练。optimizer/cos10/cache重置。

IQ256/fs25MHz，原complexRMS center=false位置不变；训练fc2.462GHz/固定正式2.45GHz，residual/post_sync/EQoffG1/ref30/qualityadaptation＋phase tracking＋IQcompensation/reflect继承。保存实际__file__/resolved args/信道值。新finalreader仅接受本run,row,final10,2220更新,精确父20/角色/默认头，不能用原As30代替新结果。

## 判定、固定评价与成本

E50源条件在formal前固定：AS_PAOFF原生fullV final student clean≥NOAS_PAOFF同口径−0.1绝对百分点；原V1800×2draw pooledmid/low均值≥原As30＋0.5pp；pooledlow≥原As30。任一FAIL则关闭本固定尾段，不扫参数、换seed、扩E50。没有额外独立DG门槛。源小差异不宣称显著因果结论。

每row固定累计E30后，自动源读出与clean＋六practical冻结预测、独立truth-last评分。全部7prediction完成后才打开该row truth。固定row不因源FAIL跳过formal；其他source未齐时不把等待当技术故障、不阻断合法固定评价。源联合决定只读source文件，最终collect先source_decision再读formal分数；任何formal成绩不反馈另一健康训练、选模、参数、种子或扩训。

正式阶段比较匹配AS_PAOFF−NOAS_PAOFF，clean下降≤0.1绝对pp；同时报告原配方、全部六practical/low/mid/urban3mean、macroF1/最差RX/RXdayTX、修复/新增错误、固定原No-As30 clean正确集合保持率、默认间隔/30、源state/lock/新增噪声比、全覆盖与曲线。重复draw非modelseed；本次只有2026092701，不能称稳定双提升。low>60及可重复clean正收益尚未达到。

本轮新增两条各E10，共20modelE/4440updates；各累计30，不在本run扩50/70。复用两E30历史60modelE/13320updates已含两个父20；全矩阵唯一训练80modelE/17760updates。native学生/教师/验证、源读出各12600、formal各1176002、真实CP/合成smoke、影子IQ数值计算与wall按实际报告，估计不当实际。

## 执行

run=20261018-phase1-FcrossRX-PAwaveform-tail-e30-s2026092701-m2-r01；owner=codex/root/PA-waveform60；CWD=PROJECT/releases/cvs_FcrossRX_PA_waveform_tail_E30_20261018_r01；命令：`CVS-RFFI/python -u experiments/cvs_pa_waveform60/runner.py`。代码/config提交、一次P0/P1、聚焦负测、复合真实父CP无query smoke、实际双臂6L合成检查、一次N607预检/归档SHA/compile与启动后PID/CWD/argv/GPU/loggrowth按最小生命周期记录；无额外审批/签名/数据重验证。

新run/release不覆盖、唯一dispatcher。每GPU≤2总PID计其他owner，free≥12000MB；另一聊天健康stepALT和所有其他任务不停止/热改/重启。协议/来源/运行错配、非有限数、输出冲突等仅停止故障所属row并保留，另一健康row继续。技术失败不自动补训练，科学负收益不算技术故障。实际commit/GPU/验证在发布时补齐。

历史证据：[旧PA-bypass](<E:/type10-7/automation_reports/CV-SincNet/20261016-phase1-F-crossRX-PAbypass-sourceRX1-e20-s2026092701-m2-r01/report.md>)；[当前普通PA审计](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-urban-prior-e30-s2026092701-m1-r01/ordinary_augmentation_audit.md>)；[P59结果](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-urban-prior-e30-s2026092701-m1-r01/final_analysis.md>)。

本地实际验证：11项focused检查PASS4.338s、13文件compile；实现者累计合成NN3840（成功3072＋一次错误toy断言768），唯一独立审查另1536NN，P0/P1=0/0、11项PASS4.340s；合计5376。原NoAs控制mixed为半ordinary/半raw，已修正测试断言而非训练实现。分析19项toy PASS.893s，NN0。真实IQ/CP/target/optimizer均0。一次普通N607预检已核实两个父20metadata/roles/参考all7完成、固定句柄复用、新root缺省及资源；尚未执行真实CP smoke或训练。

实际发布：代码提交23f794296a11e1a4f25837e78b60f36bc700b99e已push且独立GitHub compare identical；单次归档SHA比较VERIFIED、远端15py compilePASS。真实两来源无query smoke4NN/2CP/0IQ/0更新PASS后已直接训练；dispatcher1249294、NOAS worker1249382/GPU0、AS worker1249461/GPU1的PID/CWD/argv/CUDA/nvidia占用独立核实。两步日志由193700→452520、162128→356638字节，均增长；实际信道configurations与各自原控制完全一致，source_roles EXACT_MATCH/target_contact false，AdamW空state/resetcos10/lr.0002/WD.0001。当前普通PA原开关true/p_pa.14保留，新shadow实现__file__与真实leo_practical路径已核实；这不代表关闭失败，也不代表性能成功。尚无完整源/正式结果。
