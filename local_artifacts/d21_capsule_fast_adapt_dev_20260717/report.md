# D21 K10/new5胶囊轻型适应与注册开发结果

状态：`DEVELOPMENT_QUERY_DIAGNOSTIC_COMPLETED`。本轮实际使用rx20-1、seed713101、K10、5个真实新TX和三种互斥LEO_weak场景。它是开发seed机制筛选，不是独立确认矩阵或目标达成声明。

## 1. 输入与执行边界

- 输入胶囊：`E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5`
- 旧类6个，新类5个；每场景每类10个support和20个query。
- predictor只读取胶囊内已接收LEO_weak IQ、registered class index和`sealed_feature_runtime.pt`；不读取truth sidecar。
- 所有候选使用三场景K10 support LOO统一选择超参数；query不参与拟合、选择、回滚或门限确定。
- 最终predictor先写`predictions_k10_new5_l5_final.npz`，SHA256为`f8c47aeb26565e9eafdb05efa8332b8270f750f00e88bdb33de7f5591bb58820`；之后独立`score`命令才连接truth。
- 每个query独立对全部当前注册类执行argmax，无角色Oracle、类别配额、真实batch类别数、Hungarian/OT或全局重排。

## 2. 候选定义与support-only锁定值

| candidate | 机制 | support-only锁定值 |
|---|---|---|
| L0 | 单中心cosine prototype | `gamma=0,bias=0` |
| L1 | class radius bias | `gamma=-0.02` |
| L2 | radius+保守新类bias | `gamma=-0.02,new_bias=0.02` |
| L3 | radius+新类bias+每新类最近旧类稀疏边界 | `gamma=-0.02,new_bias=0.01,rival_beta=0.5` |
| L4 | 每类最多2个确定性球面prototype+最近旧类稀疏边界 | `gamma=0,new_bias=0,rival_beta=0.5` |
| L5 | 同一received IQ的`normalize(concat[z_id,8*FFT96])`+每类top1 support cosine | 无可调超参数 |

L2/L3候选同时预登记了classwise old-support invasion guard，但support LOO最终选择`guard_scale=0`；这说明当前开发support上的全局小bias比按类侵入惩罚更稳定。

## 3. 三场景真实query结果

表内依次为注册前旧类、注册后旧类、旧类floor、新类、新类floor、H和遗忘；全部来自同一row、同一旧类query配对。

| candidate | scenario | old before | old after | old floor | seen-new | new floor | H | forgetting |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L0 | clear | 0.6750 | 0.4000 | 0.0500 | 0.5600 | 0.0000 | 0.4667 | 0.2750 |
| L0 | low-elev | 0.5750 | 0.3833 | 0.1000 | 0.5100 | 0.0000 | 0.4377 | 0.1917 |
| L0 | rain | 0.7000 | 0.4417 | 0.1500 | 0.5400 | 0.0000 | 0.4859 | 0.2583 |
| L1 | clear | 0.6750 | 0.4250 | 0.1000 | 0.5500 | 0.0000 | 0.4795 | 0.2500 |
| L1 | low-elev | 0.5750 | 0.3917 | 0.1000 | 0.5600 | 0.0000 | 0.4609 | 0.1833 |
| L1 | rain | 0.6917 | 0.4667 | 0.1000 | 0.5600 | 0.0000 | 0.5091 | 0.2250 |
| L2 | clear | 0.6750 | 0.5000 | 0.1000 | 0.5500 | 0.0000 | 0.5238 | 0.1750 |
| L2 | low-elev | 0.5750 | 0.4417 | 0.1000 | 0.5100 | 0.0000 | 0.4734 | 0.1333 |
| L2 | rain | 0.6917 | 0.5083 | 0.1500 | 0.4800 | 0.0000 | 0.4938 | 0.1833 |
| L3 | clear | 0.6750 | 0.5333 | 0.1000 | 0.5200 | 0.0000 | 0.5266 | 0.1417 |
| L3 | low-elev | 0.5750 | 0.4083 | 0.1000 | 0.5400 | 0.0000 | 0.4650 | 0.1667 |
| L3 | rain | 0.6917 | 0.5000 | 0.1500 | 0.4800 | 0.0000 | 0.4898 | 0.1917 |
| L4 | clear | 0.6583 | 0.5500 | 0.0500 | 0.3500 | 0.0000 | 0.4278 | 0.1083 |
| L4 | low-elev | 0.5833 | 0.4833 | 0.0500 | 0.4600 | 0.0500 | 0.4714 | 0.1000 |
| L4 | rain | 0.6417 | 0.4667 | 0.0500 | 0.4600 | 0.0000 | 0.4633 | 0.1750 |
| L5 | clear | 0.9167 | 0.8417 | 0.7000 | 0.8700 | 0.7000 | 0.8556 | 0.0750 |
| L5 | low-elev | 0.7500 | 0.6667 | 0.4000 | 0.7600 | 0.6500 | 0.7103 | 0.0833 |
| L5 | rain | 0.8333 | 0.6833 | 0.4000 | 0.6900 | 0.5500 | 0.6867 | 0.1500 |

## 4. 三场景聚合结果

| candidate | old before | old after | old floor | seen-new | new floor | H | forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 | 0.6500 | 0.4083 | 0.1167 | 0.5367 | 0.0167 | 0.4638 | 0.2417 |
| L1 | 0.6472 | 0.4278 | 0.1000 | 0.5567 | 0.0167 | 0.4838 | 0.2194 |
| L2 | 0.6472 | 0.4833 | 0.1167 | 0.5133 | 0.0167 | 0.4979 | 0.1639 |
| L3 | 0.6472 | 0.4806 | 0.1167 | 0.5133 | 0.0167 | 0.4964 | 0.1667 |
| L4 | 0.6278 | 0.5000 | 0.1333 | 0.4233 | 0.1500 | 0.4585 | 0.1278 |
| L5 | 0.8333 | 0.7306 | 0.5167 | 0.7733 | 0.6333 | 0.7513 | 0.1028 |

逐类完整结果保存在`score_k10_new5_l5_final.json`。L0最差新类是`18-10`，聚合准确率仅0.0167；L4将聚合new floor提升到0.15但牺牲new均值。L5把所有三场景new floor提升到0.55以上，`18-10`聚合准确率达到0.9167；当前主要floor转为旧类`14-7`，聚合0.5167，low-elev/rain场景旧类floor均0.40。

## 5. 资源审计

| candidate | trainable params | epoch | after state bytes | enrollment MAC | query classifier MAC | classifier ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 0 | 0 | 7,040 | 17,600 | 1,760 | 0.000087 |
| L1 | 0 | 0 | 7,128 | 17,600 | 1,760 | 0.000094 |
| L2 | 0 | 0 | 7,148 | 17,600 | 1,760 | 0.000089 |
| L3 | 0 | 0 | 7,208 | 17,600 | 1,765 | 0.000086 |
| L4 | 0 | 0 | 14,248 | 352,000 | 3,525 | 0.000193 |
| L5 | 0 | 0 | 112,640 | FFT96单列 | 28,160 | 0.000547 |

共同的TorchScript backbone保持冻结。L5的FFT96只作用于已接收的同一IQ，每个物理support/query计算一次，不增加物理样本或K。三场景batch下，backbone约0.0213–0.0311ms/sample，FFT96约0.0999–0.1095ms/query，L5分类约0.000547ms/query；严格单query backbone调用均值5.443ms、P95 10.194ms。实测峰值CUDA显存56,691,712B（约54.1MiB）。L5当前FP32 support码状态112,640B；按逐向量int8量化的下一版主体预计28,160B，另加少量scale。全部候选均为`EVAL_ONLY_CLOSED_FORM_ADAPTATION`，0训练参数、0epoch、无dense query图，状态低于256KB。

## 6. 结论与下一机制

1. L1半径归一化只带来小幅均值改善，不能修复floor。
2. L2/L3把聚合遗忘由24.17pp降至约16.4–16.7pp，但本质仍是在旧/新之间移动决策边界，无法恢复`18-10`判别信息。
3. L4把聚合new floor从1.67%提高到15%、old floor从11.67%提高到13.33%，遗忘降到12.78pp，证明局部多模态表征方向有效；但其new均值下降且场景floor仍为0。
4. L5确认固定received IQ的FFT96局部信息是当前最大增益来源：相对L0，聚合old提高32.22pp、new提高23.67pp、H提高28.75pp，old/new floor分别提高40.00pp/61.67pp，遗忘减少13.89pp。
5. 下一轮应围绕L5压缩：先对256维融合码做int8，再验证每类2个局部prototype或“中心+最难边界码”；同时对旧类`14-7`做support-only局部边界强化。仅继续扩大bias网格没有机制依据。

## 7. 复现命令

```powershell
conda run -n ssr-gpu python -m py_compile E:\type10-7\github_publish\CVS-RFFI-repo\local_artifacts\d21_capsule_fast_adapt_dev_20260717\run_capsule_fast_adapt_dev.py
conda run -n ssr-gpu python E:\type10-7\github_publish\CVS-RFFI-repo\local_artifacts\d21_capsule_fast_adapt_dev_20260717\run_capsule_fast_adapt_dev.py predict --capsule E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5 --output E:\type10-7\github_publish\CVS-RFFI-repo\local_artifacts\d21_capsule_fast_adapt_dev_20260717\predictions_k10_new5_l5_final.npz
conda run -n ssr-gpu python E:\type10-7\github_publish\CVS-RFFI-repo\local_artifacts\d21_capsule_fast_adapt_dev_20260717\run_capsule_fast_adapt_dev.py score --prediction E:\type10-7\github_publish\CVS-RFFI-repo\local_artifacts\d21_capsule_fast_adapt_dev_20260717\predictions_k10_new5_l5_final.npz --truth E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\scorer\truth_sidecar.json --output E:\type10-7\github_publish\CVS-RFFI-repo\local_artifacts\d21_capsule_fast_adapt_dev_20260717\score_k10_new5_l5_final.json
```

验证：`py_compile`通过；L5 predict和score均退出0；最终prediction SHA为`f8c47aeb...bb58820`。L5的三场景结果与独立并行实现逐项一致。

## 8. L5-int8、L6与L7继续优化

在L5正路线确认后，同一脚本继续加入三个部署候选，均未改变IQ、K、query权限或逐样本决策方式：

- `L5q`：每个已归一化256维support code按向量对称int8量化，并保存FP16 scale。
- `L6q`：固定20epoch、256参数对角metric；loss为统一support prototype CE与K≥2 support LOO CE各0.5，变换后support code继续采用int8+FP16 scale，对角scale以FP16部署。
- `L7q`：Stage2-B先拟合`theta_B`；Stage2-C从`theta_B`初始化，加入top-2 class-CVaR、old-old pair保持和old-support新类侵入损失。三场景support LOO统一8点小网格锁定`beta=0.2,lambda_pair=0.1,lambda_inv=0.2,margin=0.01`。

### 8.1 聚合结果

| candidate | old before | old after | old floor | seen-new | new floor | H | forgetting | after state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L5 FP32 | 0.8333 | 0.7306 | 0.5167 | 0.7733 | 0.6333 | 0.7513 | 0.1028 | 112,640B |
| L5q int8 | 0.8306 | 0.7278 | 0.5167 | 0.7667 | 0.6167 | 0.7467 | 0.1028 | 28,380B |
| L6 FP32 | 0.8889 | 0.7778 | 0.6167 | 0.7900 | 0.5167 | 0.7838 | 0.1111 | 113,664B |
| L6q int8 | 0.8889 | 0.7778 | 0.6167 | 0.7900 | 0.5167 | 0.7838 | 0.1111 | 28,892B |
| L7q old-lock int8 | 0.8889 | 0.7889 | 0.5833 | 0.7933 | 0.5167 | 0.7911 | 0.1000 | 28,892B |

L5q相对L5仅下降old 0.28pp、new 0.67pp、H 0.46pp，状态减少74.8%。L6q与L6在当前330个真实query旧/新样本上逐项相同，说明int8 support code是当前部署默认。L7q相对L6q提高old 1.11pp、new 0.33pp、H 0.73pp并减少遗忘1.11pp，但聚合old floor下降3.33pp；因此L7q是均值/遗忘Pareto候选，L6q仍是floor候选。

### 8.2 L6q/L7q三场景结果

| candidate | scenario | old before | old after | old floor | seen-new | new floor | H | forgetting |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L6q | clear | 0.9250 | 0.8500 | 0.7000 | 0.8400 | 0.6000 | 0.8450 | 0.0750 |
| L6q | low-elev | 0.8500 | 0.7417 | 0.5000 | 0.8000 | 0.4000 | 0.7697 | 0.1083 |
| L6q | rain | 0.8917 | 0.7417 | 0.5000 | 0.7300 | 0.5500 | 0.7358 | 0.1500 |
| L7q | clear | 0.9250 | 0.8667 | 0.7500 | 0.8400 | 0.5500 | 0.8531 | 0.0583 |
| L7q | low-elev | 0.8500 | 0.7667 | 0.4500 | 0.8000 | 0.4000 | 0.7830 | 0.0833 |
| L7q | rain | 0.8917 | 0.7333 | 0.5000 | 0.7400 | 0.6000 | 0.7367 | 0.1583 |

### 8.3 训练与资源证据

L6q/L7q均为256参数、固定20epoch，低于50k参数和20epoch上限。每query对角变换增加256MAC，top1 support cosine为28,160MAC；当前适配相似度主项估算61,952,000MAC。L6q与L7q持久状态均为28,892B，其中int8 support code+FP16逐向量scale为28,380B，FP16对角scale为512B。

- L6 prototype+LOO最终prediction：`predictions_k10_new5_l6proto_int8_final.npz`，SHA256=`51f91d8b3173fc306287dbcdefdd673bcd6ff711409ac3b8f61a4815838da1e7`。
- L6完整loss trace：`predictions_k10_new5_l6proto_int8_final.loss_trace.json`，SHA256=`31042181afc8d12604c09fe5f0af29f39a8b9d762428d581e5cc660b637582ef`。
- L7最终prediction：`predictions_k10_new5_l7_final.npz`，SHA256=`5ee86fbecbe32173e1b2a1aa5a9720baa90ae70a8ac2c7c7d397e71ddbb1e801`。
- L7完整loss trace：`predictions_k10_new5_l7_final.loss_trace.json`，SHA256=`be37049a7b6b6b23b71ee12ee6f4bdfd97248552654c4ce46e2de30aa6fd6221`。

所有loss trace逐epoch包含总loss、LOO CE、prototype CE、support LOO accuracy/floor；L7另含CVaR、old pair MSE、old invasion loss及old/new support LOO accuracy。最终score分别保存在`score_k10_new5_l6proto_int8_final.json`和`score_k10_new5_l7_final.json`，评分只在prediction写入后读取truth。

### 8.4 L8无状态聚合消融

L8在L6q完全相同的int8 support code与对角metric上比较统一`alpha*top1+(1-alpha)*class_mean`，`alpha∈{0,0.25,0.5,0.75,1}`仅由三场景support LOO按最差逐类floor、H排序。锁定结果为`alpha=1.0`，即退化回纯top1；真实query指标与L6q逐项完全相同，class mean融合没有改善floor。该路线不再扩展。

- L8 prediction：`predictions_k10_new5_l8_final.npz`，SHA256=`0aebc62a35f1a2cc1b64699ecfa22147a62d7188aaf99471ad4a1d6a292ea885`。
- L8 score：`score_k10_new5_l8_final.json`。
- 结论：L6q保留为当前floor候选；L7q保留为均值/遗忘Pareto候选；L8 mean融合被support-only选择拒绝。

## 9. M2低秩度量实验

M2在A0的256维`z160+8×FFT96`注册表征上加入`diag(exp(theta))+UV^T`等价残差变换。统一support-only网格为`rank∈{4,8}`、`residual_reg∈{0.01,0.05}`；Stage2-B只用旧类support拟合，Stage2-C从B状态初始化并用全部已注册support拟合。固定20epoch，loss同时包含all-class LOO/prototype CE、top-2 class-CVaR、旧类pair保持和新类侵入hinge。三场景support-only选择锁定`rank=4,residual_reg=0.01`，参数量2,304。

### 9.1 真实query结果

| scenario | old before | old after | old floor | seen-new | new floor | H | forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear | 0.9500 | 0.8917 | 0.7500 | 0.8800 | 0.7500 | 0.8858 | 0.0583 |
| low-elev | 0.8083 | 0.7583 | 0.4000 | 0.7600 | 0.4500 | 0.7592 | 0.0500 |
| rain | 0.8500 | 0.6917 | 0.4000 | 0.6700 | 0.4000 | 0.6807 | 0.1583 |
| aggregate | 0.8694 | 0.7806 | 0.5500 | 0.7700 | 0.6333 | 0.7752 | 0.0889 |

相对L6q，M2将聚合遗忘降低2.22pp，new floor提高11.67pp，但old floor下降6.67pp、seen-new下降2.00pp、H下降0.86pp；它是“遗忘/new floor”方向的Pareto诊断，并未超过L6q的综合结果。low-elev和rain的old floor仍只有0.40，是当前主要失败项。

### 9.2 资源与证据

- 变换参数：2,304；固定20epoch；每query变换2,304MAC；top1分类before/after分别15,360/28,160MAC。
- 持久状态before/after：20,088B/32,988B，含int8变换后support code、FP16逐向量scale及FP16低秩参数。
- 适配MAC估算：67,020,800；分类器实测均值0.000707ms/sample、场景P95上界0.001118ms/sample。
- prediction：`predictions_k10_new5_m2_final.npz`，SHA256=`3e5814c1ef92a6865a975a492ede5a0f6bb2e8eb70159e4341f9b2bb1cc63520`。
- 完整逐epoch loss：`predictions_k10_new5_m2_final.loss_trace.json`，SHA256=`06c759ec8ec1258c13c0ae25eed566ed1f6b1134294e438d0c8421a014426394`。
- 独立评分及逐类/逐场景结果：`score_k10_new5_m2_final.json`。

验证：脚本在`ssr-gpu`绝对Python解释器下`py_compile`通过；predict与score均退出0；predictor输入不含truth/role/quota，真实标签只在不可变prediction落盘后由scorer读取。

## 10. M4：历史D1机制在current capsule上的合法复跑

旧D1结果因曾跨场景复用观测而不能作为当前协议性能证据，本节不继承旧数值。M4只读取当前胶囊中每个物理样本唯一的LEO_weak received IQ，并从该同一观测确定性计算A1：`z160+FFT96+4×RF32`。Stage2-B仅用旧类support训练共享288维对角尺度和6×288旧类权重；Stage2-C从B初始化，再仅用全部已注册support训练共享尺度和11×288类权重。两阶段各固定20epoch，query从未进入拟合、选择或回滚。

### 10.1 真实query结果

| scenario | old before | old after | old floor | seen-new | new floor | H | forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|
| clear | 0.7583 | 0.5917 | 0.3500 | 0.7400 | 0.5500 | 0.6576 | 0.1667 |
| low-elev | 0.7250 | 0.6500 | 0.3500 | 0.7200 | 0.2500 | 0.6832 | 0.0750 |
| rain | 0.7583 | 0.5583 | 0.2000 | 0.7200 | 0.5500 | 0.6289 | 0.2000 |
| aggregate | 0.7472 | 0.6000 | 0.3833 | 0.7267 | 0.4500 | 0.6573 | 0.1472 |

M4明显弱于L6q/L7q/M2：类权重head在support上收敛很高，但真实query泛化差。三场景after-support最终准确率为94.55%/98.18%/93.64%，support floor为80%/90%/80%；对应真实query old仅59.17%/65.00%/55.83%，说明该D1机制在当前单观测胶囊上发生support过拟合。它不能晋升，只保留为合法负证据。

### 10.2 资源、loss与隔离证据

- 参数：Before 2,016，After 3,456；固定20epoch；FP16持久状态Before 4,032B、After 6,912B。
- 每query总head计算：Before 2,016MAC、After 3,456MAC；无dense query图。After适配前向MAC估算7,603,200。
- head分类实测均值0.000238ms/sample，场景P95上界0.000513ms/sample；峰值CUDA allocated 64,986,112B。
- prediction：`predictions_k10_new5_m4_d1_current_final.npz`，SHA256=`bdd5317a7a4a613bfa528d56ba679a401ab3ca5930edfeed74a21f8c6da0bc34`。
- 完整loss：`predictions_k10_new5_m4_d1_current_final.loss_trace.json`，SHA256=`207ea6788cfc72649f8ab80aa13e7a0069103389c6bfbb76fde0322bc906778a`；每个场景B/C均恰好20条epoch记录。
- 独立评分与逐类/逐场景结果：`score_k10_new5_m4_d1_current_final.json`。
- predictor artifact成员仅含token/scenario/prediction与schema，未出现truth/role/quota成员；评分命令在prediction落盘后才打开truth sidecar。

最终结论：M4/D1 current-capsule路线资源最轻，但性能显著退化，`M4_D1_CURRENT_CAPSULE_NO_GO`。本轮到此停止，不扩展新机制。
