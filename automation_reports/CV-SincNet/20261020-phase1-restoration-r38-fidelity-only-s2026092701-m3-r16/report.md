# R16：F-free纯IQ教师压缩与逐场景0.8pp保持

状态：ANALYZED。用户最新验收为相对固定原R38 dense_half完整794529参数，每场景性能最多下降0.8个百分点。正式七视图各168000物理样本，最多减少1344条正确预测，整数500×正确数损失≤4×样本数，0均值替代。原R17另列参考，0改变历史实验验收口径。

唯一直接父模型为原R38 source E20/4940、method e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381；继承同7/63/30和已核实原R17源祖先，0继承R13/R14/R15已测学生，非新独立盲测。训练仍0TX标签、0F权重/输出、0V/target优化，只合法L_s/U_s IQ、原配对clean和合法源IQ教师。

源侧假设：精确原父的teacher-stage NMSE为0，但paired-clean目标通常非零，其梯度会推动模型偏离父函数。此前两目标相加既要拟合父又要改写父。R16只优化三个恢复stage对原父IQ输出的平均NMSE；clean IQ只作为原有能量归一化和detached诊断，不作为重建优化项。此选择不依赖F/V分类分数或目标结果。新损失、矩阵、预算、seed、rank与最终轮次在target访问前固定。

|行|完整参数|可训练prior|压缩结构|GPU|
|---|---:|---:|---|---:|
|factor625_fidelity|750035|201240|prior卷积SVD 0.625|2|
|factor500_fidelity|720185|171390|prior卷积SVD 0.5|3|
|dense750_fidelity|689129|140334|GN4 prior宽度.75|6|

原548795参数coarse/action逐位冻结，student与teacher共享一次不可变context，部署仍自带完整context。SVD仅在减少参数时转换，left/right权重跨三个stage共享、bias独立；dense行沿既有GN4切片，不新增IQ校准。所有prior均可训练。

三行固定E10/2470、L63000/U567000曝光、teacher/studentprior各630000、共享context630000/studentcontext0；received01..10、MODELseed2026092701/DATAseed2026102001，reset AdamW1e-4/warm247/cos1e-5/WD1e-4/clip1、B256/micro64真实尾24。context累计40、prior累计50、component-epoch130、祖先串行pass90、F50E50另计。教师/context/F/head零更新。

每版自动source1800×7诊断/profile，固定E10后均target168000×clean、三suburban、三urban；每版全部七视图预测固定后独立truth-last，原F50/defaulthead不重训。报告原R38精确0.8pp、实际升降、净纠错、TX/RX/day、clean损伤、波形主体/尾部/FFT/配对clean特征，lowurban70与其他五场景距rawclean5pp单列。0源分数门槛、0目标反馈调参或选择性重跑，所有负结果保留。

部署只当前原生float32[B,2,256] IQ，完整自身context/prior/3×2CG，无teacher、clean、标签、场景或状态更新。完整参数、Conv/Linear MAC（FFT/CG另列）、文件字节、16sourceV/batch1/2warmup实测完整链延迟/显存一并报告，不凭减参推断加速。

GPU2/3/6各一own/max3，2/3原R15、6原R14对应owned句柄终态且实际退出才交接，free>=18GB、全owner<2，其他owner保持，原R06共享锁max1own scorer/CPU2。新不可覆盖run/release；普通预检、一次归档两端SHA、远端compile。launcher首步真实父/F50/2Ls no-query smoke通过立即训练；prior梯度、精确冻结context、teacher/F/head零更新、严格重载输出一致。技术故障保留该行、健康行继续，0自动重试/回退。

REJECTED_EXTRA_GATE：项目八项最小流程覆盖通用设计/计划审批，一次独立P0/P1限于下一真实运行。具体命令、环境、CWD、路径、停止规则与artifact见experiment.json。

真实发布VERIFIED：method 4626eb3e378d2d975c875516f75e0fecd09647a2，归档两端一次SHA一致、远端compile通过；launcher真实权重smoke PASS。GPU2/3/6实际PID、CWD、完整命令、commit、GPU及首轮日志增长逐项核实，其他owner不变。18项聚焦检查和唯一独立P0/P1审查PASS；报告计算入口12项聚焦检查PASS。尚未以源结果宣布正式识别成功。

## R16 完整结果交付

三行均完成纯教师IQ目标固定 E10/2470、source 1800×7 和 target 168000×7；每行全部预测固定后独立 truth-last。实际所属进程已退出，三份完整恢复器已取回并严格加载。全部负结果保留，结果不回流训练或重跑。

|版本|参数|较 R38 减少|最大下降 R38/R17（pp）|low urban（%）|R38/R17 各自 0.8pp|训练 F 知识|
|---|---:|---:|---:|---:|---|---|
|factor625_fidelity|750035|5.6000%|2.939881/1.886905|65.329762|False/False|False|
|factor500_fidelity|720185|9.3570%|3.822619/2.769643|64.447024|False/False|False|
|dense750_fidelity|689129|13.2657%|3.663690/2.610714|64.605952|False/False|False|

进一步缩小且通过用户原79万参考0.8pp的 IQ 行：无；使用冻结 F 知识的行：无。本轮三版均为严格 F-free；只更新 prior，coarse/action 保持并冻结。


既有10轮参考：factor625对R15同秩、dense750对R14同宽，只有优化目标不同；factor500对R14原宽度，结构和目标均不同，不能单因素归因。0.8pp 通过仅表示满足退化容差。三版训练均不加载 F，不使用 TX 标签、V 或 target。low urban≥70% 和其他五场景距 raw clean≤5pp 分别记录，不以压缩验收替代。单 seed、已暴露基准，不称独立盲测或全局最小。

|版本|场景|准确率（%）|相对 R38（pp）|相对 R17（pp）|相对既有10轮参考（pp）|净纠错|
|---|---|---:|---:|---:|---:|---:|
|factor625_fidelity|clean|84.817262|+0.187500|+0.230357|-0.237500|+237|
|factor625_fidelity|practical_high|81.182738|-0.964881|-0.136310|+0.260714|+67956|
|factor625_fidelity|practical_mid|79.585119|-1.388095|+0.007143|+0.069643|+66643|
|factor625_fidelity|practical_low_suburban|77.960714|-1.532738|-0.035714|-0.021429|+65404|
|factor625_fidelity|practical_high_urban|79.916071|-1.239286|-0.437500|+0.175000|+67566|
|factor625_fidelity|practical_mid_urban|72.633929|-2.142262|-1.051190|-0.249405|+64093|
|factor625_fidelity|practical_low_urban|65.329762|-2.939881|-1.886905|-0.449405|+58851|
|factor500_fidelity|clean|85.513690|+0.883929|+0.926786|+0.938095|+1407|
|factor500_fidelity|practical_high|81.684524|-0.463095|+0.365476|-0.116667|+68799|
|factor500_fidelity|practical_mid|80.108929|-0.864286|+0.530952|-0.444643|+67523|
|factor500_fidelity|practical_low_suburban|78.341071|-1.152381|+0.344643|-0.763690|+66043|
|factor500_fidelity|practical_high_urban|80.227976|-0.927381|-0.125595|-0.602976|+68090|
|factor500_fidelity|practical_mid_urban|72.183333|-2.592857|-1.501786|-2.404762|+63336|
|factor500_fidelity|practical_low_urban|64.447024|-3.822619|-2.769643|-3.670833|+57368|
|dense750_fidelity|clean|85.623810|+0.994048|+1.036905|+0.592857|+1592|
|dense750_fidelity|practical_high|81.953571|-0.194048|+0.634524|-0.213095|+69251|
|dense750_fidelity|practical_mid|80.441667|-0.531548|+0.863690|-0.253571|+68082|
|dense750_fidelity|practical_low_suburban|78.572619|-0.920833|+0.576190|-0.306548|+66432|
|dense750_fidelity|practical_high_urban|80.520238|-0.635119|+0.166667|-0.220833|+68581|
|dense750_fidelity|practical_mid_urban|72.328571|-2.447619|-1.356548|-0.473810|+63580|
|dense750_fidelity|practical_low_urban|64.605952|-3.663690|-2.610714|-0.486905|+57635|

|版本|coarse/action/prior 参数|Conv+Linear MAC/IQ|文件字节|R38/本版中位 ms|resident/增量 peak 字节|
|---|---|---:|---:|---:|---:|
|factor625_fidelity|329782/219013/201240|78298688|3324004|43.1058/49.2368|11783168/317162496|
|factor500_fidelity|329782/219013/171390|71501888|3204516|59.3583/64.3685|11653632/317162496|
|dense750_fidelity|329782/219013/140334|67956800|3042844|44.0803/46.9946|11550208/317188608|

MAC 不含 FFT、复数 CG、归一化、激活、插值和访存。实测包含完整恢复链，使用 16 条 source V、batch 1、2 次预热；物理求解保持 3 个恢复阶段、每阶段 2 次 CG，共 6 次 CG 迭代。顺序和共存负载影响延迟，不从参数比例推断加速。原 R17 的同口径延迟和峰值显存保留在 results.json。


### factor625_fidelity：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.048190205→0.026073994；loss 0.017067072→0.0020781943；teacher_stage_nmse 0.017067072→0.0020781943。中位每轮 128.199 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.000675969932|0.000679095974|0.000605069101|0.0509027094|0.0009323804988525808|
|practical_high|0.0183631536|0.00805729162|0.11884734|0.296678632|0.016107939183712006|
|practical_mid|0.0235430207|0.0135390526|0.120873258|0.34018451|0.020243102684617043|
|practical_low_suburban|0.0291021168|0.019379247|0.123422652|0.375985742|0.02473633922636509|
|practical_high_urban|0.0234949291|0.0134120686|0.121486574|0.313936353|0.019257426261901855|
|practical_mid_urban|0.0482300185|0.0392441712|0.13405399|0.412138522|0.03622397407889366|
|practical_low_urban|0.0710159764|0.0628608242|0.147227377|0.484701812|0.0535648837685585|

### factor500_fidelity：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.055597598→0.029331558；loss 0.022792269→0.0041627586；teacher_stage_nmse 0.022792269→0.0041627586。中位每轮 131.728 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.00100765261|0.000971045811|0.00129191577|0.0652951151|0.004041743464767933|
|practical_high|0.0185874924|0.00816214178|0.120565176|0.306548655|0.015758154913783073|
|practical_mid|0.0239630286|0.0138754509|0.12245623|0.3522093|0.020381953567266464|
|practical_low_suburban|0.0297345202|0.0199591536|0.124892786|0.389425039|0.02532951720058918|
|practical_high_urban|0.0239072144|0.013674017|0.123740174|0.324408561|0.01934143155813217|
|practical_mid_urban|0.0495948903|0.04037771|0.138156697|0.426274121|0.038285840302705765|
|practical_low_urban|0.0733320788|0.064872995|0.152978614|0.501675665|0.057072993367910385|

### dense750_fidelity：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.051307487→0.028521035；loss 0.019861545→0.0036050904；teacher_stage_nmse 0.019861545→0.0036050904。中位每轮 114.955 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.000791480183|0.000803944829|0.000640717917|0.0543455184|0.002450346015393734|
|practical_high|0.0184684526|0.00812163763|0.119265825|0.306710839|0.016999106854200363|
|practical_mid|0.0237638354|0.0137192952|0.121421918|0.352905363|0.021466584876179695|
|practical_low_suburban|0.029531166|0.0197652876|0.124181874|0.392484665|0.026365669444203377|
|practical_high_urban|0.023691155|0.0135197416|0.122467794|0.325828165|0.020472029224038124|
|practical_mid_urban|0.0489586294|0.0397394486|0.136962131|0.432252645|0.03902650251984596|
|practical_low_urban|0.0726926997|0.0641641691|0.152368575|0.512263|0.05760611966252327|

逐 TX/RX/day 的准确率、Macro-F1、纠错/损伤、混淆矩阵、配对 clean 几何与完整训练曲线见 results.json 和 row_metrics.csv。这些指标不能证明恢复了全部真实 TX 细节。coarse/action 各累计 40 轮，prior 累计 50 轮；祖先训练共 90 串行 pass，component-epoch 合计 130，F50 预训练另计。

本地模型：local_artifacts/pamf_restoration/r38_fidelity_only_20261020_r16/models/<arm>/final.pt。加载 experiments.pamf_r38_fidelity_only.runtime.load_checkpoint；输入原生尺度 float32 [B,2,256] IQ，输出同形 IQ，接原 F50 默认头。权重不入 Git。目标状态依据全部识别指标保留，不以本轮完成宣称长期优化目标完成。
