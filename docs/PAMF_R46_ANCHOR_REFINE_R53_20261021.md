# R53：持续 IQ 教师约束的压缩性能优化

状态：LOCAL_VERIFIED，未发布。性能优先；不把更小参数、容差通过或源域小幅涨分当作正式性能优化完成。

研究差异来自完整源域目标审计：IQ 重建误差改善可能伴随固定识别器源准确率下降。原衰减1→0.1教师约束可能不足以保持父模型的逐输入响应；这里只检验持续权重1的波形教师约束，不用TX标签、F输出或测试分数设置新损失。该假设可能无效，全部负结果保留。R52两个相位参考均不足，未据此施加C4等变或不变损失。

固定三行：prune_decay273765保留线性衰减；retain_anchor350521和prune_anchor273765教师系数恒为1。其余仍是原pairedclean IQ NMSE＋0.05多尺度相对功率STFT＋三stage IQ教师NMSE，coarse/action/三个prior全部联合训练，原3prior/6CG/GN4结构和输入输出不变，无额外推理分支或参数。两宽度均来自已登记R48完整两点矩阵，直接父继续唯一为人指定R46 prune_refine350521 E20/4940（bbb8daf817de050b958d217b70d4c33f4aefe341），不加载R48/R49派生学生。

同宽350521原衰减控制复用R49 retain_refine，micro256、seed、数据、预算相同，不重复训练。27.38万衰减控制本次新训以匹配micro256；R48同宽micro64结果保留，仅作历史/吞吐对照。原八行全部正式测试已完成。所有新行固定最终E20/4940，无源选模或提前结束，训练后立即source1800×7、同卡G/G+F成本、正式168000×7预测；每row七预测固定后独立CPU truth-last评分，不因其他行未训完等待整批，也不据结果调参、重排或选择性重跑。

原6300L_s＋56700U_s全部使用，TX标签对恢复器隐藏；原7/63/30物理划分/R07缓存epoch01..20、model2026092701、split392005、data2026102001、augmentation2026101912、formal2027均不变。每行新增20轮/4940更新/global256micro256/直接float32buffer/ResetAdamW，LR2e-4暖247至2e-5、WD1e-4、clip1、FP32/TF32关闭。每行L126000/U1134000曝光、1260000学生和父教师IQ，读取TX/F/V/target训练信息为0。继承每组件80轮，新增20轮后每组件100、组件和300、串行累计140；F50预训练成本另计。

源source_cache READY/EXACT_MATCH和R46/R44/R38/R17/R07/R04源祖先契约复用；R46checkpoint来源已核实原source-only final，F50默认头仅训练完成后读取评价，不另训头。属于有clean模拟配对的TX无标签自监督，不能称无clean盲学习。正式基准已暴露，保留工程比较背景，不声称新的独立盲测或逐细节TX真实性。

按当前两卡约束仅物理GPU2/3，各一个本任务worker；prune_decay和retain_anchor并行，prune_anchor在GPU2排队，GPU worker最多2。每行CPU2，共享原score.lock最多1个本任务scorer。2026-10-21只读资源检查全卡空闲；实际一次N607 preflight和启动后PID/CWD/argv/GPU/日志另存。新不可覆盖release/run/output，唯一owner codex/root/R46-anchor-refine，其他owner及历史产物保持。低准确率不触发技术停止；直接权限/输出碰撞/错误checkout或split/无法合法预测等系统技术失败保留受影响输出、无自动重试/热改/回退，不停止健康同伴。

32项聚焦检查与CLIhelp通过，唯一一次限P0/P1审查见实际review.json；首步真实R46/F50/2Ls配对IQ noquery smoke，PASS立即训练。Git commit固定代码/config；一次预检、单归档两端SHA比较和compile，禁止额外设计审批/签名/成员哈希等门槛（REJECTED_EXTRA_GATE）。精确命令、环境/CWD、路径、三行矩阵与预期产物以experiment.json为准。

正式主要指标是七景accuracy/Macro-F1、相对R17/R46的逐景差、修复/损伤与逐TX/RX/day；原R17逐景最多下降1pp为退化容差，不是性能优先的成功定义。low urban≥70%、其他五受损场景距rawclean84.6761904762%≤5pp以及更强TX细节真实性目标继续ACTIVE。成本次级：同卡64源V，G和G+F的B1/B32三轮交替同步计时、median/P90/P95、吞吐、峰值/增量/共享驻留显存、完整参数/序列化大小/ConvLinear MAC；不含传输/前处理/loader，MAC不含FFT/CG/norm/激活/F，驻留是3G＋1F共享suite，不冒充单模型显存。
