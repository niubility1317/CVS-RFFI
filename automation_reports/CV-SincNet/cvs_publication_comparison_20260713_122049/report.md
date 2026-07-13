# CVS论文级方法对比实验Git交接

- 根报告：`E:\type10-7\automation_reports\CV-SincNet\cvs_publication_comparison_20260713_122049\report.md`
- 追踪表：`analysis/cvs_publication_comparison_traceability_20260713.md`
- 统一协议：`docs/CVS_PUBLICATION_COMPARISON_PROTOCOL_20260713.md`
- 当前阶段：Phase1 seed713101三个baseline仍在训练；Stage2-C K5 seed713101中CSIL和MoPC-HR已完成，Orthogonal仍在运行；Stage2-B三种监督DA的低步数smoke已完成。
- Stage2-C已实现CSIL、MoPC-HR、Orthogonal Incremental统一runner、三种LEO测试、固定query的seeded nested K-shot split、sample score及四层详细统计。正式K5锚点已完成的CSIL/MoPC-HR分别得到`H_old_new=0.1808/0.2459`，单seed且Orthogonal未完成，禁止作正式排序。
- Stage2-B已实现ProtoNet CDA、MRIOR-SDA、DADDA-SDA统一监督runner。每个run只适应一个target receiver，仅有标签target-old support可训练；query只评估。三方法smoke均输出360条score、57条四层明细且全测试星地增强。
- 监督DA聚焦回归`10 passed`；新增runner提交`c157754`。N607同步hash、remote py_compile和dry-run均通过。
- 声明边界：smoke只证明机制与artifact契约；Phase1终局详细后评估、Stage2-C完整K/seed矩阵、Stage2-B五接收机K/seed矩阵和CVS同协议结果仍未完成，因此不构成论文最终性能结论或部署成功证据。

## CVS同协议入口与训练预算决议（13:50）

- Phase1当前CVS候选不是普通监督基线：`phase1_dgleo_jointp0_leoweak8r2_20260713`以`ADV3B02_CORE90_SOFT_E200`为初始化，使用`rho_label=0.08`有标签与0.72源域无标签，包含三种星地信道训练视图和source-val-only选择。论文表必须显式报告其额外无标签访问预算，不能把它写成仅算法结构差异。
- Phase2提出方法固定为两个与项目谱系一致的入口：Stage2-B使用冻结CVS特征上的`CVS-OPGAC`监督support-only原型高斯校准；Stage2-C使用冻结`ADV3B02_CORE90_SOFT_E200`特征上的`CVS-qKNNV42`。后者参数固定为int8 support code、类内top-1、prototype权重0.45、old anchor0.001、8轮support-clamped label propagation权重0.025；unknown拒识不进入Phase2主线。
- 新增`paper_reproduction/cvs_aligned/cvs_method_runner.py`，强制单target receiver、三种正式LEO缓存、`seeded_nested` K={1,2,5,10,20}、support pool maxK=20后固定query、query标签不训练/不选模、sample score和四层明细、有限support-fit trace。
- Stage2-B与Stage2-C正式矩阵均从3个对比方法扩为“CVS+3个对比方法”，每阶段为`4方法 x 5接收机 x 5K x 5seed=500`行。新dry-run manifest分别生成500行。
- 训练预算采用双层报告：主表使用CVS任务下的common-budget，以控制训练计算量并保证同一数据、K、seed、receiver和query配对；论文原生epoch/batch只作为方法谱系敏感性附表，不作为主表直接混排。该选择避免Orthogonal的100/50 epoch与CSIL/MoPC短训练配置造成计算预算不等，但主表必须标为CVS extension而非论文原始结果。
- 本地验证：`cvs_method_runner.py`与matrix worker py_compile通过；新增CVS runner与matrix聚焦测试`4 passed`。当前尚未生成三场景ADV3B02正式feature cache，也未启动500行矩阵。

## ADV3B02正式feature cache导出计划（13:48）

- 目标：为CVS-OPGAC与CVS-qKNNV42生成同一冻结`ADV3B02_CORE90_SOFT_E200`、同一原始样本集合的三份星地场景feature cache；每份包含clean source、五个target receiver的target-old与两个target-new TX。
- 本地脚本：`paper_reproduction/scripts/export_cvs_publication_adv3b02_features_20260713.sh`；本地`bash -n`和`--dry-run`通过。导出只做冻结模型前向，不训练、不更新checkpoint。
- checkpoint：`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；特征`z_id`；ManySig old TX索引0-5；ManyTx new TX=`1-16,1-18`；target receivers=`20-1,3-19,7-14,7-7,8-8`；target day0；每TX最多400条。
- 三个并行短任务分别使用GPU3/4/5，输出`runs/cvs_publication_adv3b02_feature_cache_20260713/{leo_clear_weak,leo_low_elev_weak,leo_rain_weak}.npz`，日志位于`paper_reproduction/logs/cvs_publication_adv3b02_feature_cache_20260713/`。
- 精确服务器入口：`bash paper_reproduction/scripts/export_cvs_publication_adv3b02_features_20260713.sh`，环境`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，工作目录`/home/szu2070436088/2510044040/CV-SincNet`。
- 成功条件：三份NPZ非空；target-old/new行的`sat_scenarios`分别全部等于对应场景；五个receiver每个old/new TX至少包含40条以满足maxK20+query20；三场景sample ID集合一致；checkpoint hash和manifest可读取。任一条件失败则不启动CVS正式矩阵。

### feature cache完成与CVS K5锚点计划

- 导出PID3796469正常完成；三份NPZ均5300行、160维，target-old/new无unknown/proxy行。正式验证`PASS`：三场景target行均只含对应星地场景；五receiver每个old/new TX均满足maxK20+query20覆盖；三场景共享完全相同的5300个sample ID。
- 本地回收：`local_artifacts/cvs_publication_adv3b02_feature_cache_20260713/`。SHA256：clear=`c5639f3c...a867`、low-elev=`912219b0...a872`、rain=`8f0b4e58...eb1a`。
- 下一步先跑receiver20-1、K5、seed713101的CVS-OPGAC与CVS-qKNNV42正式锚点，输出根`paper_reproduction/runs/cvs_publication_cvs_anchor_k5_seed713101_20260713/{cvs_opgac,cvs_qknnv42}`。两者使用同一三场景cache；OPGAC只登记target-old support，qKNNV42登记target-old+target-new support；query标签不进入适应或选模。
- 精确命令为`python -m paper_reproduction.cvs_aligned.cvs_method_runner --config paper_reproduction/configs/cvs_proposed_stage2_publication_features_n607.json --run-dir <method_dir> --method <cvs_opgac|cvs_qknnv42> --target-receiver 20-1 --seed 713101 --split-seed 713101 --k-shot 5 --device cpu`。成功条件为8个artifact、三场景、四层明细、finite trace、support/query无重叠及全测试星地增强。

### CVS K5锚点v1审计与修正

- v1两方法artifact链路PASS：OPGAC为360条score/57条明细/3条finite trace，qKNNV42为480条score/78条明细/3条finite trace；两者旧类support逐ID一致，均无重叠且全测试星地增强。
- v1数值：CVS-OPGAC适应前0.6722、适应后0.7361、delta+0.0639；CVS-qKNNV42 old0.6056、seen-new0.6333、H0.6170、forgetting0.0861。
- 反向审计发现v1 qKNNV42遗漏技术报告中固定的`diag_whiten_fisher` support-only变换（strength0.1），因此v1 qKNNV42降级为实现诊断，不能进入主表。已补入严格support-only的类间/类内Fisher对角缩放与对角whitening；三场景中每个已登记类都有同场景support，故`scenario_residual_weight=0.5`按公式为零并显式记录，而不是静默省略。
- 修正后py_compile及CVS runner+matrix测试仍为`4 passed`。需同步后写入新根`cvs_publication_cvs_anchor_k5_seed713101_v2_20260713`，不得覆盖v1。

### CVS K5锚点v2与方法原生优化器修正（14:05）

- qKNNV42修正版锚点已完成并回收到`local_artifacts/cvs_publication_cvs_anchor_k5_seed713101_v2_20260713/`。同一receiver20-1、K=5、seed713101下，CVS-OPGAC保持适应前0.6722、适应后0.7361、delta+0.0639；CVS-qKNNV42得到old=0.6417、seen-new=0.6500、H_old_new=0.6436、forgetting=0.0667。
- v2两方法均含完整8件artifact、三种正式LEO场景、四层详细统计、sample-level score与finite loss trace。v1 qKNNV42因遗漏support-only变换继续仅作诊断，不进入主表；v2仍只是单receiver、单K、单seed锚点，不能据此给出正式排序。
- 为避免“统一步数”被误写成“统一优化器”，Stage2主表预算明确为`common_steps_method_native_optimizer`：相同base/adapt或increment步数，但保留各方法原生优化器与关键超参数。Stage2-C已分别设置CSIL SGD(lr0.01,momentum0.9,wd0.01)、MoPC SGD(lr0.01,momentum0.9,wd0.0002)、Orthogonal base/increment SGD(lr0.01/0.08,momentum0.9,wd0.0005)，并恢复Orthogonal noise0.01、top-k60、tau-fuse0.01、embedding256、pseudo-targets200。
- Stage2-B中MRIOR-SDA采用本地论文方程复现谱系的Adam、lr0.0006、wd0；DADDA-SDA采用论文配置的SGD、momentum0.9、wd0.0005、base/adapt lr0.0001及`(1+10p)^-0.75`反比衰减。ProtoNet CDA保留既有ProtoNet训练入口。主表仍须标为CVS extension；MRIOR论文未报告优化器，因此Adam属于已披露的本地复现选择，不得声称论文原文指定。
- 本地聚焦回归：`conda run -n ssr-gpu python -m pytest tests/test_cvs_supervised_da_runner.py tests/test_cvs_class_incremental.py tests/test_cvs_publication_matrix.py tests/test_cvs_proposed_stage2_runner.py -q`，结果`19 passed`。完整500行矩阵在方法原生trace smoke通过前不得启动。

### 方法原生trace smoke启动前记录（14:08）

- 本地版本：Git提交`6f09dd2`；同步文件为`class_incremental.py`、`supervised_da_runner.py`及Stage2-B/C正式配置。远端SHA256分别为`e0062f40...a3`、`e8853388...36`、`5e89e68...aa`、`df007515...02`，与本地一致；remote py_compile通过；Stage2-B/C各500行manifest的4行dry-run通过。
- 服务器状态：Phase1 baseline仍各占GPU0/1/2一个进程，显存约448/528/502MiB；GPU3-7空闲。计划在GPU3-7各启动一个短smoke，不触碰Phase1，不超过每GPU两个训练进程。
- 输出根：`paper_reproduction/runs/cvs_publication_method_native_trace_smoke_20260713/`；日志根：`paper_reproduction/logs/cvs_publication_method_native_trace_smoke_20260713/`；工作目录`/home/szu2070436088/2510044040/CV-SincNet`；Python`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- Stage2-B命令模板：`CUDA_VISIBLE_DEVICES=<3|4> python -m paper_reproduction.cvs_aligned.supervised_da_runner --config paper_reproduction/configs/cvs_stage2b_supervised_da_publication_base_n607.json --run-dir <mrior_sda|dadda_sda> --experiment-id <method>_native_trace_smoke --method <method> --target-receiver 20-1 --seed 713101 --split-seed 713101 --k-shot 5 --base-steps 5 --adapt-steps 3 --device cuda:0`。
- Stage2-C命令模板：`CUDA_VISIBLE_DEVICES=<5|6|7> python -m paper_reproduction.cvs_aligned.class_incremental --config paper_reproduction/configs/cvs_stage2c_publication_base_n607.json --run-dir <csil|mopc_hr|orthogonal_incremental> --experiment-id <method>_native_trace_smoke --method <method> --target-receiver 20-1 --seed 713101 --split-seed 713101 --k-shot 5 --base-steps 5 --old-support-steps 3 --increment-steps 3 --device cuda:0`。
- 这些短步数结果只验证方法原生优化器、有限loss trace和artifact契约，不作为性能排序。成功条件：进程退出0、8件artifact齐全、三种LEO场景、support/query不重叠、finite trace，并在resolved config/metrics中记录方法原生配置。

### 方法原生trace smoke完成与完整矩阵启动计划（14:15）

- 五个smoke均退出0：MRIOR-SDA PID3803644、DADDA-SDA PID3803645、CSIL PID3803646、MoPC-HR PID3803647、Orthogonal PID3803648。每方法8件artifact齐全、support/query overlap=false、三种正式LEO场景、finite trace；Stage2-B各360条score/57条明细，Stage2-C各480条score/78条明细。
- 优化器审计PASS：MRIOR base/adapt均为Adam lr0.0006、wd0；DADDA base/adapt均为SGD momentum0.9、wd0.0005，trace从lr0.0001衰减至0.000016556；Stage2-C resolved config记录`common_steps_method_native_optimizer`及全部方法原生关键参数。smoke数值不进入性能排序。
- Phase1只读状态：CVCNN在epoch199/200，RIEI在epoch198/200，DRIFT在epoch76/200；三个进程仍分别位于GPU0/1/2，不干预。DRIFT中间星地rain约20.37%只是epoch74附近诊断，不是终局结果。
- 新增五分片resume-safe launcher：`paper_reproduction/scripts/launch_cvs_publication_stage2_matrix_20260713.sh`；本地`bash -n`与Stage2-B dry-run通过。每阶段500行按5个shard映射GPU3-7，输出分别为`cvs_publication_stage2b_full_matrix_20260713`与`cvs_publication_stage2c_full_matrix_20260713`。
- 计划同时执行`bash paper_reproduction/scripts/launch_cvs_publication_stage2_matrix_20260713.sh stage2b --execute`和`bash paper_reproduction/scripts/launch_cvs_publication_stage2_matrix_20260713.sh stage2c --execute`。每GPU恰有两个矩阵worker，符合每GPU最多两个训练实验的项目上限；任一已有同shard进程会fail closed。正式完成条件是每阶段500行均通过artifact contract，之后才聚合同receiver/K/seed的配对统计。

### 完整Stage2矩阵已landed（14:17）

- Stage2-B五个worker：shard0-4分别为PID3806411/3806413/3806417/3806420/3806423，对应GPU3-7。
- Stage2-C五个worker：shard0-4分别为PID3806427/3806430/3806434/3806437/3806441，对应GPU3-7。
- 启动后只读复核：10个worker均存活；GPU3-7显存约742/742/970/1012/742MiB，未超过每GPU两个实验；GPU0-2的Phase1进程保持原状。各阶段manifest为500行，每shard分配100行；worker逐行执行并在完整artifact contract通过后才记录complete，失败则该shard停止。
- 启动脚本远端SHA256=`557b28c3...ed2`，remote `bash -n`及两阶段dry-run通过。此处只证明landed并开始执行，不等于artifact-complete或论文结果完成。

### Phase1详细星地后评估启动前记录（14:25）

- CVCNN-CE与RIEI-FD训练已完成并生成`best_by_val.pt`及`metrics.json`；DRIFT仍在GPU2训练，保持monitor-only。Stage2-B/C均无失败，继续占用GPU3-7。
- 新增`paper_reproduction/scripts/evaluate_cvs_phase1_detailed.py`，严格加载validation选择的`best_by_val.pt`，只评估三个正式星地场景及三个main OOD split；输出sample score、overall、split、receiver、transmitter、receiver-transmitter、receiver-transmitter-day统计及混淆信息。clean不进入正式结果。
- 数据协议与训练完全一致：ManySig、equalized=1、train ratio0.1、val ratio0.9、guard gap8、train day0/1、test day2/3、train receiver0-6、test receiver7-11、seed713101。星地随机种子与训练内置评估一致：sat seed2027及scenario/split确定性偏移。
- 本地`py_compile`与聚合单测通过。计划分别在GPU0/1运行CVCNN/RIEI详细后评估，输出到各训练run的`detailed_satellite_eval/`；DRIFT完成后在GPU2运行同一入口。后评估不会修改checkpoint或训练结果。

### Phase1详细星地后评估已landed（14:28）

- CVCNN-CE详细评估PID3817352、GPU0；RIEI-FD详细评估PID3817353、GPU1。两者工作目录、Python环境和数据协议均与训练记录一致，日志分别为`cvcnn_ce_detailed_satellite_eval.log`和`riei_fd_detailed_satellite_eval.log`。
- 启动后进程与显存复核正常，无Traceback；此时仍在完整三场景前向，尚未生成终局`metrics.json`，因此只记为landed，不记为artifact-complete。
- Stage2监控点：Stage2-B已完成72/500、失败0；Stage2-C已完成9/500、失败0，五个Orthogonal正式行仍在运行。DRIFT到epoch101/200。

### CVCNN-CE与RIEI-FD详细Phase1结果完成（14:31）

|方法|checkpoint epoch|LEO clear|LEO low-elev|LEO rain|sample rows|detail rows|结论|
|---|---:|---:|---:|---:|---:|---:|---|
|CVCNN-CE|101|21.5147%|20.7667%|20.9946%|612000|894|artifact PASS|
|RIEI-FD|85|20.6515%|20.2760%|20.4554%|612000|894|artifact PASS|

- 两方法详细结果均覆盖3场景×204000条测试样本，具备overall、per-split、per-receiver、per-transmitter、per-receiver-transmitter、per-receiver-transmitter-day六层统计；正式结果不含clean，全部经过星地信道。
- 后评估总体正确数与训练时validation-selected checkpoint的内置星地评估逐场景完全一致，证明详细统计没有改变样本集合、随机星地扰动或checkpoint。
- 本地回收路径：`local_artifacts/cvs_publication_phase1_detailed_seed713101_20260713/{cvcnn_ce,riei_fd}/`；本地复核score row分别为612000、detail row分别为894。当前Stage2-B为92/500、失败0；Stage2-C为9/500、失败0；DRIFT仍在训练并于epoch104触发新的validation-best测试。

### CVS未来接收机快速行加速计划（14:35）

- GPU0/1已完成Phase1详细后评估并空闲；GPU2继续DRIFT，GPU3-7保持完整矩阵worker。为避免长期闲置且不改变训练方法预算，仅提前执行未来四个receiver的无训练CVS feature-cache行：Stage2-B `cvs_opgac`和Stage2-C `cvs_qknnv42`。
- 加速receiver限定为`3-19,7-14,7-7,8-8`，K/seed保持完整矩阵默认网格，各100行；输出根与正式矩阵相同，独立event log根为`.../accelerator_cvs_rows`。正式主worker到达时通过artifact contract跳过已完成行。
- 不加速需要训练的ProtoNet/MRIOR/DADDA/CSIL/MoPC/Orthogonal，避免与现有worker未来相撞。精确入口为matrix worker的`--methods cvs_opgac`或`--methods cvs_qknnv42`、`--receivers 3-19,7-14,7-7,8-8`、`--execute`，分别使用GPU0/1。

### CVS快速行完成与manifest隔离修正（14:37）

- Stage2-B CVS加速PID3823666：assigned100、completed98、skipped2、failed0；Stage2-C CVS加速PID3823667：assigned100、completed100、failed0。按完整默认网格现场重建后，Stage2-B总artifact完成209/500，Stage2-C完成109/500。
- 发现subset worker的shard0会覆盖canonical `matrix_manifest.json`。该问题不影响已启动main worker的内存行列表或任何run artifact，但当前远端canonical manifest暂时只反映subset，不能作为500行完成证据。
- 本地已修正matrix worker：完整默认网格继续写`matrix_manifest.json`，任何方法/receiver/K/seed子集改写到基于选择哈希的`matrix_manifest_subset_<hash>.json`，新增防覆盖回归；聚焦回归`20 passed`。
- 按active-job monitor-only边界，当前不热补远端worker文件。待现有完整worker退出后同步修正并用完整默认dry-run重建500行canonical manifest；完成度在此期间仅通过`build_rows(DEFAULT_*)`逐行执行artifact contract现场计算。

### 论文统计聚合入口预落地（14:45）

- 新增`paper_reproduction/scripts/summarize_cvs_publication_stage2.py`，最终会严格按完整默认网格逐行执行artifact contract；任一行不完整时默认拒绝生成正式论文汇总。
- 预定义输出包括：1000行run-level结果、3000行scenario-level结果、method×K均值/标准差/95%CI、method×receiver×K五seed统计、同receiver/K/seed配对的CVS差值及win/tie/loss摘要。Stage2-B参考为CVS-OPGAC，Stage2-C参考为CVS-qKNNV42。
- 统计单位以每个receiver-seed run的三场景均值为主，避免把同一run的三个星地场景错误当作三个独立重复；scenario-level表单独保留用于场景敏感性分析。聚合脚本py_compile及单测通过。
- 当前现场完成度：Stage2-B 220/500、Stage2-C 115/500、失败0；DRIFT到epoch126/200。聚合入口只在全部行完成并同步到远端后执行正式模式。

### Phase1 CVS详细对比入口与声明边界（14:52）

- Phase1 CVS对比行固定为本轮低标签半监督联合P0族中heldout原始性能最佳的`JP0_J5_U_TRI_STRONG/final_ssdg.pth`：epoch80、checkpoint hash=`e21dee17...af8`。其现有冻结heldout证据为LEO clear79.5235%、low-elev76.9642%、rain76.4608%，均为204000条main OOD样本。
- 该J5行是`NON_PROMOTABLE_DIAGNOSTIC`：内部sat strict floor70.4933%未达到CVS promotion目标，且训练使用约0.08有标签+0.72源域无标签、ADV3B02 teacher初始化；因此只能作为CVS原始对比性能并明确披露额外无标签和teacher访问，不能写成同等监督数据预算或部署成功。
- 新增`paper_reproduction/scripts/evaluate_cvs_phase1_ssdg_detailed.py`，严格从checkpoint args重建SSDG模型和split，固定三个正式LEO场景、sat seed2027及三个main OOD split，输出与baseline同结构的612000条sample score和receiver/transmitter/day六层统计；checkpoint missing/unexpected key必须为0。
- 本地py_compile和入口测试通过。因当前N607存在活跃训练，按monitor-only边界暂不同步/启动；待现有矩阵worker退出后与manifest修正、统计聚合器一起同步，再执行J5详细后评估并用既有heldout逐场景正确数做一致性验收。
- 最新现场进度：Stage2-B 235/500、Stage2-C 126/500、失败0；DRIFT到epoch127附近。

### 最终完成性审计器预落地（14:58）

- 新增`tools/validate_cvs_publication_comparison.py`作为最终硬门：Phase1必须同时具备CVS-J5、CVCNN-CE、RIEI-FD、DRIFT四方法的612000条score和894条六层明细；Stage2-B/C必须各500行、每方法125行且逐行artifact contract通过；canonical manifest必须与完整实验ID集合完全一致；论文汇总必须为1000条run-level、3000条scenario-level且incomplete=0。
- 审计同时强制三个正式LEO场景、`all_tests_satellite_augmented=true`、clean不进入正式结果，并要求per-receiver、per-transmitter和receiver-transmitter-day等详细层级完整。任一条件缺失即退出非零且生成带错误列表的审计JSON。
- 本地py_compile及聚焦测试`5 passed`。当前现场完成度为Stage2-B 252/500、Stage2-C 129/500、失败0；DRIFT在epoch129后的validation-best完整星地评估中。
- 审计器与前述两个待同步入口一样，仅在活跃worker全部退出、canonical manifest修复和论文聚合完成后同步并执行；目前不能用本地测试替代最终远端artifact审计。
