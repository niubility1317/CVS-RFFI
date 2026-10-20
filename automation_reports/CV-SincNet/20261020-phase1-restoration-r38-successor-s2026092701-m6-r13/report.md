# R38 79万参数基底的继续压缩与优化 R13

状态：ANALYZED。用户最新明确在79万参数模型基础继续压缩优化。本轮唯一直接父权重为R38 dense_half、完整794529参数、固定source E20/4940、commit e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381；原R17为合法source祖先。用户在已暴露benchmark后指定此基底，如实披露，不称新独立盲测。R38训练/固定最终权重未使用TX/F/V/target优化或target选epoch；新训练不读取目标反馈。既有 R11/R12 六行已完成并归档。

|固定行|完整/可训练参数|context参数|相对794529减少|GPU|训练F知识|
|---|---:|---:|---:|---:|---|
|keep_iq|794529|548795|0.0000%|2|False|
|three_quarter_iq|442805|302471|44.2682%|3|False|
|half_iq|204561|140315|74.2538%|6|False|
|keep_task|794529|548795|0.0000%|7|True|
|three_quarter_task|442805|302471|44.2682%|4|True|
|half_task|204561|140315|74.2538%|5|True|

固定3宽度×2目标，原宽度是同新增预算对照。完整R38原宽度逐位copy；其余按原GN四组等量取通道，decoder拼接使用旧完整upstream宽度后的skip索引；保留谱分支首层投影拓扑，三步prior长卷积共享、每步bias/GN/1×1独立。0IQ/TX/F/V/target校准，coarse/action/prior全部联合训练，教师保留自身完整R38 context并冻结。三步×2CG保持。

所有行损失：最终paired source-clean IQNMSE +1×三个stage到冻结R38输出的IQNMSE均值，分母为detach clean能量/floor1e-4。task行额外T1KL(Fclean||Frestored)+0.5单位F160距离平方；IQ控制正式训练0F权重加载/前向，不能只把权重置零仍执行F。原F50/defaulthead0参数更新、0头拟合。task使用原监督F知识，不称严格无监督；全部0新TX/V/target训练标签。

每行只新增E10/2470、L63000/U567000/source630000，R38 IQ教师/studentcontext各630000；仅task行Fclean/Fstudent各630000。原cache63000=L6300/U56700、既定received01..10、15%clean/85%residual、model2026092701/data2026102001/FP32/noTF32。B256/micro64以真实globalbatch分母累积，尾24正确归一化，每globalbatch一次clip1/update。重置AdamW，LR1e-4/warm247/cos1e-5、WD1e-4；这是独立微调日程，不声称恢复原optimizer轨迹。原祖先coarse20/action20/R17prior20 +R38joint20 +newjoint10=90串行pass；各组件累计50/component sum150，原F50预训练50轮/11100与smoke/profile另计。

固定最终E10，无早停、源分数选行或target资格门槛。每行训练后自动source1800×7诊断/profile，然后正式同168000物理样本×七目标场景；每行七视图IQ/logit/features/IDs全部固定后，独立CPU才读truth/clean及原baseline汇总。逐场景同时报告相对用户79万R38和原R17差值，二者均以100×(参考正确数−新正确数)≤N作为1pp底线；另报真实提分、lowurban≥70%、其他场景距clean、paired-clean全帧/主体232/尾24/FFT/身份特征差及TX/RX/day、净纠错与完整10轮曲线。所有固定行及负结果保留。

推理只当前原生尺度float32 [B,2,256] IQ，输出同形恢复IQ并接原F50/defaulthead；0teacher/clean/TX/RX/scene/cache/跨帧状态更新。成本比较同时含原R17、794529父版与学生：固定16条source V/batch1/2warmup，包含完整context/CG，GPU payload/optimizer/F/teacher排除；不由减参预先宣称加速。

41项合成/协议检查PASS；唯一独立P0/P1发现预测入口原固定E20与新E10不匹配，已修复并经一次定向复审PASS。共享入口默认E20保持，R13显式声明E10/2470，3个实际调用路径回归检查通过。launcher首步实际R38父权重/F50/2Ls CPU noquery smoke，检查所有组件与F输入有限非零梯度、父/F/head不变、严格复载逐位输出、0TX/V/target/optimizer更新，PASS立即逐卡排队。GPU2–7每卡一own/max6；相应R11或R12旧row终态并且实际ownedPID/cmd句柄退出后才派发新行；不等待其他卡、不抢占健康任务。共享R06 score.lock/max1own CPU scorer/2threads；源训练空闲>=18GB且全owner每卡<2进程。

新不可覆盖run/release/output、一次普通N607预检、单archive两端SHA、远端compile/help、实际controller/worker PID/CWD/cmd/commit/GPU/日志读回。队列等待只称QUEUE_VERIFIED，不冒称GPU训练已开始；GPU启动后记录E1。技术错误停止受影响行并保留、健康行继续，不自动重试/回退。具体命令、环境、CWD、路径、停止规则和预期产物见experiment.json。

普通N607预检VERIFIED：R38实际父权重source E20/4940、0TX/F/V/target训练、各组件累计40及原R17source-only来源核实；新run/release不存在。R11/R12六个实际GPU句柄仍在运行，只提交CPU真实权重smoke与逐卡接续队列；0原任务停止/重启。

实际队列发布VERIFIED：方法commit `593bf5905f50fb7894e475b062bc191a2837a149` 已推送且远端OID独立比较一致，单归档SHA与远端compile/help通过。实际CPUcontroller PID/CWD/完整命令/commit及持续等待日志核实；真实R38最终权重/F50/2Ls noquery smoke PASS，六行全部组件及F输入有限非零梯度、父/F/head0更新、严格复载逐位输出一致。当前旧R11/R12六卡继续健康运行，新行按各卡真实句柄退出接续；排队状态不等于GPU训练或目标性能已验证。

GPU 实际训练 VERIFIED：keep_task (GPU7, E1), half_iq (GPU6, E1)。实际 PID/CWD/命令/commit/GPU 与完整首轮记录核实，0 新技术故障；其余行按原队列接续。目标结果尚未齐全，不宣称完成。

GPU 实际训练 VERIFIED：three_quarter_iq (GPU3, E1), keep_iq (GPU2, E1)。实际 PID/CWD/命令/commit/GPU 与完整首轮记录核实，0 新技术故障；其余行按原队列接续。目标结果尚未齐全，不宣称完成。

GPU 实际训练 VERIFIED：half_task (GPU5, E1), three_quarter_task (GPU4, E1)。实际 PID/CWD/命令/commit/GPU 与完整首轮记录核实，0 新技术故障；其余行按原队列接续。目标结果尚未齐全，不宣称完成。

## R13 完整结果交付

六行均完成固定 E10/2470、source 1800×7 和 target 168000×7；每行全部预测固定后独立 truth-last。实际所属进程已退出，六份完整恢复器已取回并严格加载。全部负结果保留，结果不回流训练或重跑。

|版本|参数|较 R38 减少|最大下降 R38/R17（pp）|low urban（%）|R38/R17 各自 1pp|训练 F 知识|
|---|---:|---:|---:|---:|---|---|
|keep_iq|794529|0.0000%|0.167857/0.000000|68.502976|True/True|False|
|three_quarter_iq|442805|44.2682%|3.937500/2.884524|64.332143|False/False|False|
|half_iq|204561|74.2538%|11.804167/10.751190|56.465476|False/False|False|
|keep_task|794529|0.0000%|2.539881/1.486905|65.729762|False/False|True|
|three_quarter_task|442805|44.2682%|1.470238/0.436310|66.799405|False/True|True|
|half_task|204561|74.2538%|14.351190/13.298214|53.918452|False/False|True|

进一步缩小且同时通过两个参考容差的 IQ 行：无；使用冻结 F 知识的行：无。两种训练权限分别解释，不能将 task 结果替代 F-free 恢复器结果。


1pp 通过仅表示满足退化容差。IQ 行训练不加载 F；task 行使用监督训练的冻结 F 知识，须单独标注。low urban≥70% 和其他五场景距 raw clean≤5pp 分别记录，不以压缩验收替代。单 seed、已暴露基准，不称独立盲测或全局最小。

|版本|场景|准确率（%）|相对 R38（pp）|相对 R17（pp）|相对同预算原宽度（pp）|净纠错|
|---|---|---:|---:|---:|---:|---:|
|keep_iq|clean|84.679762|+0.050000|+0.092857|True/True|+6|
|keep_iq|practical_high|82.102381|-0.045238|+0.783333|True/True|+69501|
|keep_iq|practical_mid|80.805357|-0.167857|+1.227381|True/True|+68693|
|keep_iq|practical_low_suburban|79.350000|-0.143452|+1.353571|True/True|+67738|
|keep_iq|practical_high_urban|81.120238|-0.035119|+0.766667|True/True|+69589|
|keep_iq|practical_mid_urban|74.837500|+0.061310|+1.152381|True/True|+67795|
|keep_iq|practical_low_urban|68.502976|+0.233333|+1.286310|True/True|+64182|
|three_quarter_iq|clean|84.741071|+0.111310|+0.154167|False/False|+109|
|three_quarter_iq|practical_high|81.077381|-1.070238|-0.241667|False/False|+67779|
|three_quarter_iq|practical_mid|79.592262|-1.380952|+0.014286|False/False|+66655|
|three_quarter_iq|practical_low_suburban|77.665476|-1.827976|-0.330952|False/False|+64908|
|three_quarter_iq|practical_high_urban|79.755357|-1.400000|-0.598214|False/False|+67296|
|three_quarter_iq|practical_mid_urban|72.126786|-2.649405|-1.558333|False/False|+63241|
|three_quarter_iq|practical_low_urban|64.332143|-3.937500|-2.884524|False/False|+57175|
|half_iq|clean|84.411310|-0.218452|-0.175595|False/False|-445|
|half_iq|practical_high|81.279167|-0.868452|-0.039881|False/False|+68118|
|half_iq|practical_mid|78.767262|-2.205952|-0.810714|False/False|+65269|
|half_iq|practical_low_suburban|75.919643|-3.573810|-2.076786|False/False|+61975|
|half_iq|practical_high_urban|79.023214|-2.132143|-1.330357|False/False|+66066|
|half_iq|practical_mid_urban|67.211905|-7.564286|-6.473214|False/False|+54984|
|half_iq|practical_low_urban|56.465476|-11.804167|-10.751190|False/False|+43959|
|keep_task|clean|84.780952|+0.151190|+0.194048|False/False|+176|
|keep_task|practical_high|81.197619|-0.950000|-0.121429|False/False|+67981|
|keep_task|practical_mid|79.719048|-1.254167|+0.141071|False/False|+66868|
|keep_task|practical_low_suburban|78.124405|-1.369048|+0.127976|False/False|+65679|
|keep_task|practical_high_urban|79.891071|-1.264286|-0.462500|False/False|+67524|
|keep_task|practical_mid_urban|72.526190|-2.250000|-1.158929|False/False|+63912|
|keep_task|practical_low_urban|65.729762|-2.539881|-1.486905|False/False|+59523|
|three_quarter_task|clean|84.150595|-0.479167|-0.436310|False/True|-883|
|three_quarter_task|practical_high|82.129167|-0.018452|+0.810119|False/True|+69546|
|three_quarter_task|practical_mid|80.818452|-0.154762|+1.240476|False/True|+68715|
|three_quarter_task|practical_low_suburban|79.081548|-0.411905|+1.085119|False/True|+67287|
|three_quarter_task|practical_high_urban|80.897024|-0.258333|+0.543452|False/True|+69214|
|three_quarter_task|practical_mid_urban|74.034524|-0.741667|+0.349405|False/True|+66446|
|three_quarter_task|practical_low_urban|66.799405|-1.470238|-0.417262|False/True|+61320|
|half_task|clean|84.319643|-0.310119|-0.267262|False/False|-599|
|half_task|practical_high|79.508333|-2.639286|-1.810714|False/False|+65143|
|half_task|practical_mid|77.534524|-3.438690|-2.043452|False/False|+63198|
|half_task|practical_low_suburban|75.404167|-4.089286|-2.592262|False/False|+61109|
|half_task|practical_high_urban|76.348214|-4.807143|-4.005357|False/False|+61572|
|half_task|practical_mid_urban|63.833333|-10.942857|-9.851786|False/False|+49308|
|half_task|practical_low_urban|53.918452|-14.351190|-13.298214|False/False|+39680|

|版本|coarse/action/prior 参数|Conv+Linear MAC/IQ|文件字节|R38/本版中位 ms|resident/增量 peak 字节|
|---|---|---:|---:|---:|---:|
|keep_iq|329782/219013/245734|88858688|3473116|42.4133/42.4061|20470784/317162496|
|three_quarter_iq|176962/125509/140334|49266224|2054300|44.5279/51.5474|19108352/317188608|
|half_iq|82454/57861/64246|22558752|1104860|43.4602/43.2092|18161152/79972864|
|keep_task|329782/219013/245734|88858688|3473116|43.6064/43.2719|20470784/317162496|
|three_quarter_task|176962/125509/140334|49266224|2054300|43.6929/50.9696|19108352/317188608|
|half_task|82454/57861/64246|22558752|1104860|44.0786/55.1452|18161152/79972864|

MAC 不含 FFT、复数 CG、归一化、激活、插值和访存。实测包含完整恢复链，使用 16 条 source V、batch 1、2 次预热；物理求解保持 3 个恢复阶段、每阶段 2 次 CG，共 6 次 CG 迭代。顺序和共存负载影响延迟，不从参数比例推断加速。原 R17 的同口径延迟和峰值显存保留在 results.json。


### keep_iq：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.024312779→0.021789959；loss 0.025337765→0.022349156；teacher_stage_nmse 0.0010249857→0.00055919646。中位每轮 171.071 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.000809152261|0.000837371801|0.000506164855|0.036864616|0.0002668218803592026|
|practical_high|0.017283069|0.00758891506|0.111270353|0.274331242|0.014183124527335167|
|practical_mid|0.0224364232|0.0129714832|0.114012219|0.312526464|0.018133603036403656|
|practical_low_suburban|0.0280696675|0.0188251305|0.117224775|0.344669163|0.022427210584282875|
|practical_high_urban|0.022532938|0.013055264|0.114090458|0.290911913|0.01718064397573471|
|practical_mid_urban|0.0470917113|0.0385901257|0.127676681|0.381540328|0.033293575048446655|
|practical_low_urban|0.0698438957|0.0620649122|0.141846851|0.448834449|0.04969017207622528|

### three_quarter_iq：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.26136707→0.031860043；loss 0.4972092→0.040559783；teacher_stage_nmse 0.23584213→0.0086997407。中位每轮 170.111 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.0016422997|0.00170461822|0.0010900012|0.0432014614|0.00047892099246382713|
|practical_high|0.0188755412|0.00810243469|0.122804992|0.30520916|0.01637805812060833|
|practical_mid|0.024331389|0.0137936287|0.125769079|0.351031542|0.0214009378105402|
|practical_low_suburban|0.0302358568|0.0199040361|0.129482567|0.391538978|0.026947440579533577|
|practical_high_urban|0.024278542|0.0137372985|0.125666097|0.324564159|0.020203804597258568|
|practical_mid_urban|0.04980281|0.0403264388|0.139525443|0.431163788|0.04034707322716713|
|practical_low_urban|0.0739245713|0.0651815683|0.154970884|0.512985826|0.0603361651301384|

### half_iq：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.41173742→0.046345999；loss 0.8118281→0.068481453；teacher_stage_nmse 0.40009067→0.022135454。中位每轮 168.319 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.00434106821|0.00445557758|0.00305201788|0.0802786723|0.001642012968659401|
|practical_high|0.0249358863|0.0125934873|0.146625012|0.3649818|0.021714694797992706|
|practical_mid|0.032364089|0.0204377621|0.149965569|0.420912802|0.02793504111468792|
|practical_low_suburban|0.0399793684|0.0284126941|0.153923735|0.47159487|0.034382108598947525|
|practical_high_urban|0.0328765661|0.0209138952|0.150517076|0.390748739|0.027874261140823364|
|practical_mid_urban|0.0676968768|0.0573214404|0.168298393|0.526745439|0.05631445720791817|
|practical_low_urban|0.099380441|0.0900287703|0.187987968|0.631318033|0.08163061738014221|

### keep_task：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.02881272→0.027889057；loss 0.18273134→0.11809884；task_KL 0.13086814→0.073204585；task_feature_distance 0.018328848→0.012639244；teacher_stage_nmse 0.0047216258→0.0043659544。中位每轮 220.437 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.000949817128|0.000975081872|0.000677862845|0.0503185615|0.0003232381131965667|
|practical_high|0.0186147112|0.00834214687|0.119068988|0.301158518|0.013945910148322582|
|practical_mid|0.0243592188|0.0143676288|0.122009769|0.346841186|0.01807851903140545|
|practical_low_suburban|0.0304891765|0.0207379349|0.125646025|0.386599392|0.022598711773753166|
|practical_high_urban|0.0245675799|0.0145322792|0.12242756|0.321357697|0.017109347507357597|
|practical_mid_urban|0.0521555468|0.0432117879|0.138258412|0.431062132|0.03343767672777176|
|practical_low_urban|0.0772471204|0.0690935552|0.154603332|0.513425887|0.049350887537002563|

### three_quarter_task：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.39765336→0.067467782；loss 1.358211→0.2614084；task_KL 0.52518377→0.13071694；task_feature_distance 0.066198447→0.023609859；teacher_stage_nmse 0.36917543→0.039613823。中位每轮 213.509 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.00321440771|0.0032108177|0.00355715863|0.0697607175|0.0007274343515746295|
|practical_high|0.028238561|0.0128667429|0.17862308|0.461791515|0.0192657969892025|
|practical_mid|0.0393483117|0.024522543|0.184548452|0.529689848|0.02381204068660736|
|practical_low_suburban|0.0510881282|0.0368677266|0.190637201|0.59634316|0.028840823099017143|
|practical_high_urban|0.0410984196|0.025876468|0.189812437|0.495747536|0.023308824747800827|
|practical_mid_urban|0.0981550962|0.084423013|0.231912643|0.673035502|0.04329196736216545|
|practical_low_urban|0.150168598|0.138200283|0.266538948|0.814475715|0.0631890520453453|

### half_task：完整训练与波形证据

全量 10 轮损失首轮→末轮：iq_nmse 0.55605169→0.14291905；loss 1.6842493→0.5806421；task_KL 0.52267479→0.28023978；task_feature_distance 0.071212529→0.040648642；teacher_stage_nmse 0.53431032→0.11683462。中位每轮 208.252 s。日志命中：[]。

|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|
|---|---:|---:|---:|---:|---:|
|clean|0.0141827492|0.0145021509|0.00813100953|0.145304829|0.0031438374426215887|
|practical_high|0.0420398973|0.0215003714|0.238859951|0.563395381|0.023692121729254723|
|practical_mid|0.0669376105|0.0474521108|0.253984004|0.636829019|0.03067072108387947|
|practical_low_suburban|0.0907478109|0.07241676|0.267395139|0.714268267|0.03735031560063362|
|practical_high_urban|0.0871535242|0.0680147111|0.270736814|0.629994631|0.035051312297582626|
|practical_mid_urban|0.260047168|0.24699454|0.387776375|0.920926273|0.07770625501871109|
|practical_low_urban|0.395780981|0.388103873|0.474572867|1.14213693|0.10768062621355057|

逐 TX/RX/day 的准确率、Macro-F1、纠错/损伤、混淆矩阵、配对 clean 几何与完整训练曲线见 results.json 和 row_metrics.csv。这些指标不能证明恢复了全部真实 TX 细节。每组件累计 50 轮；祖先训练共 90 串行 pass，component-epoch 合计 150，F50 预训练另计。

本地模型：local_artifacts/pamf_restoration/r38_successor_20261020_r13/models/<arm>/final.pt。加载 experiments.pamf_r38_successor.runtime.load_checkpoint；输入原生尺度 float32 [B,2,256] IQ，输出同形 IQ，接原 F50 默认头。权重不入 Git。目标状态依据全部识别指标保留，不以本轮完成宣称长期优化目标完成。

参考口径说明：three_quarter_task 相对原 R17 最差下降 0.436310pp，满足用户原 R17 1pp 容差；相对用户指定的 R38 父模型最差下降 1.470238pp，未通过本轮另列的 R38 保持标准。该行使用监督 F 知识，不能替代 F-free 结果。
