# F＋跨 RX：等剂量早期与后期 As 的固定 E20 源域检验

run_id：20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01。当前 PLANNED，无真实训练结果。逐行唯一矩阵见 experiment.json；固定配置和停止规则在首次运行前锁定，不按任何目标分数调整。

前一轮完整质量代理诊断已否定原整体状态切换假设。另一路逐包六场景循环已完成，较随机As仅low＋0.1587pp、urban均值＋0.5026pp，低于原门槛，固定方案关闭。实际F实例还核实没有BatchNorm（只有GroupNorm/LayerNorm、206个state张量、900722参数），所以不启动BN运行统计修补。这些事实不证明Ag或联合目标有效。

本轮唯一假设：固定预算内，将相同七轮As监督安排在E14至20，相比E1至7，能改善clean保持，并保留urban适应。它检验训练时序策略，不声称已隔离“先学身份”的唯一因果机制：学习率和既有SSL阶段也随日历改变；两组As总曝光相等，但不冒称逐包/逐draw多重集合完全相同。完整实际L序列、状态/场景/物理ID覆盖需独立报告，不按预测或损失挑包。

## 固定模型、数据与四个匹配控制

增强前基线继续F＋跨RX同TX浅/中层对比（SIG_TIME_X3、统一Sinc、真实joint160、NoExplicitCFO及原内部必要补偿），默认cos30头随骨干正常联合训练，无新部署模块/头/参数，不冻结骨干另训头。所有模型scratchE20、seed2026092701、4440更新、原13L＋7U、oldgate/ownEMA、AdamW与cosine保持。当前不是新Ag或Ag×As四因子矩阵。

复用split392005/VALIDATED_ONCE原sourcecarrier；本轮fit RX3/4/6/8，day1至3，TX0至5；L5040、U45360真值隐藏、V21600仅健康。held source RX1的原L1260仅在固定FINAL20自动读出七场景。L6300原carrier、fitL5040和历史L_train5580/L_dev720不能混用；不读target、不重验未变数据、不加载祖先权重。兼容smoke唯一例外严格载入同有效L/U/V来源已确认的B00 FINAL20，2个合成包no-query后丢弃，不作为学生或教师初始化。

完整原base＝.75raw＋.25ordinary＋原ordinary-only浅/中层CROSS_RX SupCon，绝对份额保持；另加.25RD(view)，RD为同一个已生成IQ的两次CE均值＋原RDrop KL1。每组8L前向、总pairweight1.25，无新跨视图一致性/对齐/课程门控。inactive额外raw时raw总权重1.00，active As时raw总权重.75，明确披露相对份额变化。所有合法有标签As均参与CE，不要求urban学生先高置信。

|行|第四对的As日历|实际As监督包数|每步额外前向|原base/SSL|
|---|---|---:|---:|---|
|RAW|无；全20轮用raw|0|2|保持|
|ALL|E1至20|568320|2|保持|
|EARLY7|E1至7；其余raw|198912|2|保持|
|LATE7|E14至20；其余raw|198912|2|保持|

四行每一步都从同一raw、原物理元数据和原命名随机流生成单次完整practical shadow（每行568320包），只由epoch确定是否送入第四对。没有连续两次完整信道，不对ordinary再加As，不独立放大扰动，不改变场景边际概率，不用状态/质量/TX真值定义session。原几何/遮挡/传播/噪声/接收质量及补偿依赖保留。

实际星地入口仍为既有backend的experiments/adv3b02_xuc/code/leo_practical，actual_channel.json/actual_modules.json保存实际__file__和全配置。25MHz、2.462GHz、IQ256；原cache RMS、进入信道normalize_input；residual/post_sync，EQoff时G=1，reference_snr_db=30，reflect，原相位跟踪、IQ补偿、AGC/人工边界/数值延迟保持。RX硬件参数按rx/day/eq session稳定，不由TX定义；训练流20261016101与source_eval20261016103分离，receiver20261016102。新增噪声SNR不是总SNR，不反演原录音未知TX硬件，也不改变正式clean＋六practical receivedIQ/信道。

## 首次固定判据与放弃规则

LATE7必须满足以下全部条件，同时完成全部源域读出：

- 对RAW：clean和clean宏F1不下降，low≥＋5pp，urban均值≥＋3pp。
- 对ALL：clean≥＋1pp，clean宏F1不降，low和urban均值各≥−0.5pp。
- 对等剂量EARLY7：clean≥＋1pp，clean宏F1不降，low及urban均值各≥＋1pp。
- 对三个控制的每个practical及seenV均≥−0.5pp；全4440更新/4546560LNN/3330SupCon步/额外1136640LNN、真实生成及消费计数正确，原完整base最终目标≤0.5。

FAIL关闭固定等剂量早晚时序，不改起点、剂量、轮次、系数、RX、seed或门槛救跑，不将控制事后晋级。PASS仅支持另行登记源域确认；不等于有效Ag、正式target收益或跨seed联合成功。当前run明确source-only，失败行不访问target。

## 预算、执行和技术停止

4个新模型×20＝80model-epochs、17760更新，无父训练/继承成本。每行原ordinary568320次、新As生成568320次（包含RAW和inactive shadow）；合计2273280次As生成，实际As监督共966144包。每行4546560LNN（额外1136640）＋实际原U/EMA/V/8820held；预期总NN23293274含dispatcher兼容smoke2，最终按真实计数核对，教师成本、物理生成、NN和墙钟分开报告。影子生成成本明确，不以减少输入构造成本冒充时序优势。

普通N607原CVS-RFFI环境，CPU2/worker、fullFP32，原容量策略计入所有owner/preCUDA，每GPU最多3进程、空闲至少12000MiB；不停止或修改任何健康任务。唯一owner codex/root/FcrossRX-AsTiming-20261017；CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_AsTiming_sourceRX1_E20_20261017_r01。入口：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_AsTiming_sourceRX1_E20_20261017_r01/experiments/cvs_as_timing20/runner.py。输出=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01，日志=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01/logs。本地证据=E:/type10-7/local_artifacts/cvs_FcrossRX_training_observation_20261017。

本次Git commit为实际代码/配置身份；一次N607 preflight已PASS，发布将做一个archive本地/远端SHA比对和remotecompile。launcher首步同契约真实CP no-query smoke，PASS立即继续四行scratch；实际PID/CWD/argv/GPU/log增长独立核实后才记RUNNING。角色/来源/非有限/错误预算/输出冲突/不可启动等系统技术失败保留产物、停止新派发，不自动重试，不停止健康已运行worker。普通低分不是技术失败，不热改或选择性重跑。

本地compile和15个协议反例、7个科学反例PASS；真实CVS合成检查8216NN／1024完整物理调用／4个联合更新PASS，六/十四轮激活、同选中IQ成对、四行同shadow统计、206state/900722参数、默认头更新和导出joint160均核实。真实IQ/CP/target为0；测试首个计数断言曾混淆state张量与参数张量，已在任何NN前修正，未重跑真实实验。一轮限定独立P0/P1静态审查PASS。

## 自动收尾和交付边界

每行FINAL20后即自动冻结checkpoint内原默认头，读出clean＋六种原practical的held source预测、margin/30和传播/接收元数据，不等其他row完成后才测试。随后独立全量复算各row/TX/day/状态/质量/最差RX、宏F1、paired修复/新增错误、固定RAW clean正确集的sat保持率、原L/U顺序、实际覆盖、loss/梯度曲线和成本。单held RX1不能冒充五个RX最差值，四row也不是四个seed。所有负结果和缺项保留。

真实IQ、checkpoint、逐包ID/预测、原日志、生成图保留私有产物；Git仅代码/config/登记/报告与合规聚合。正式目标评分、本轮默认头收益/clean代价、跨seed重现以及low>60%/suburban接近自身clean均尚未证明；总目标保持ACTIVE。
