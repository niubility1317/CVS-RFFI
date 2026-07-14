# qKNNV42星上轻量化计算剖析与首轮压缩报告

## 1.实验信息

|字段|内容|
|---|---|
|experiment ID|`qknnv42_lightweight_head_20260714_173701`|
|时间|2026-07-14 17:37:01+08:00|
|操作方|Codex|
|目标|压缩qKNNV42星上计算资源，并在相同receiver、seed、K-shot和LEO场景下验证`old_acc`、`seen_new_acc`、`H_old_new`相对原qKNNV42下降均不超过3pp|
|当前阶段|首轮head压缩与FFT96独立确认完成；5-view自适应压缩待下一轮|
|对照|2026-07-13单视图、无FFT、dense label propagation的125次`cvs_qknnv42`正式Stage2-C结果|

## 2.协议与部署边界

- 冻结`ADV3B02_CORE90_SOFT_E200`，星上不做full-model fine-tuning。
- target receiver与source receiver不相交；每行同时包含6个target-old TX和2个seen-new TX。
- support/query均来自同一target receiver的`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`视图，且互不重叠。
- 本轮只优化Phase2 Stage2-C的旧类适应和seen-new注册识别，不把unknown拒识作为主目标。
- 历史5-view、60 epoch adapter和角色/类别配额Hungarian仅作为资源与oracle诊断参照；角色Oracle不属于可部署输入。

## 3.首轮计算剖析结论

|排序|模块|主要复杂度|星上影响|首轮处理|
|---:|---|---|---|---|
|1|5-view TTA+ADV3B02前向|约为单视图backbone前向的5倍；FFT也重复5次|端到端算力和能耗最大|已有单视图125-run；等待同切分完整5-view分支完成后做严格性能差比较|
|2|dense query label propagation|构造`(S+Q)×(S+Q)`全矩阵，8轮传播；FFT双分支时执行两次|内存随batch平方增长，不适合流式逐样本推理|本轮新增可关闭模式，优先移除该最重qKNN head模块|
|3|support-memory qKNN|约`Q×S×D`余弦积|随K和注册类数线性增长|后续评估prototype压缩，不在本轮提前牺牲局部邻居信息|
|4|96维FFT辅助|每视图一次256点FFT，并使qKNN打分链执行第二遍|明显小于5次backbone，但会增加前处理和head计算|保留到后续消融；先验证去dense graph的零/低精度损失路径|
|5|Hungarian角色/类别配额|分块三次复杂度；依赖query角色与配额Oracle|大batch下非流式，且输入不可部署|正式轻量路径固定`per_sample_argmax`，不启用Oracle|
|6|60 epoch`id_norm_late_feature`|地面训练成本；推理时复用同一backbone结构|不属于星上在线训练预算，但增加部署准备成本|不作为首轮星上推理head压缩对象|

## 4.需求追踪

|ID|来源|需求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|QK-L01|用户目标|量化各模块计算、内存、延迟并从最重项开始|本报告、benchmark artifact|`pending`|模块级复杂度表和实测延迟|端到端5-view严格对照等待N607完整分支|
|QK-L02|用户目标|实现可部署的轻量qKNN head|`paper_reproduction/cvs_aligned/cvs_method_runner.py`|`pending`|单元测试、125-run|默认行为保持向后兼容|
|QK-L03|用户目标|相对原性能下降不超过3pp|125-run配对结果|`pending`|同receiver/seed/K/scenario的`old/new/H`配对差|三个主指标分别检查|
|QK-L04|`项目.md`|保持Stage2-C的`R_t/R_s`、`Y_old/Y_new`和LEO视图协议|配置、split manifest|`pending`|125/125 artifact、support/query overlap=0|不更改科学协议|
|QK-L05|星上部署约束|去除query角色Oracle与batch quota依赖|runner配置与manifest|`pending`|`per_sample_argmax`、`role_oracle_used=false`|dense LP关闭后还需`query_used_for_transductive_inference=false`|
|QK-L06|报告纪律|保存逐run同row指标、资源统计和结论|本报告、artifacts|`pending`|125行完整汇总|不使用跨行拼接最大值|

## 5.当前证据

原单视图无FFT、dense LP的125次对照已完整：`old_acc=65.5933%`、`seen_new_acc=47.9400%`、`H_old_new=53.2562%`；平均head计时`34.665 ms/run`，平均`0.2167 ms/query`。单视图FFT96的另一个125-run为`old_acc=75.1244%`、`seen_new_acc=64.6400%`、`H_old_new=68.5647%`，平均`0.1693 ms/query`；该计时不包含raw IQ backbone和FFT导出，因此只能用于head内部比较。

N607上的完整60 epoch adapter+5-view+FFT96+legacy Oracle分支仍在运行；2026-07-14 17:35只读检查显示训练到epoch 60并开始导出，尚无125个`metrics.json`。任务运行期间保持monitor-only，不启动、终止或修改远端作业。

## 6.版本与验证记录

- 根目录`E:\type10-7`不是Git仓库。
- Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`。
- 编辑前分支：`codex/cvs-rffi-release-20260626`，相对远端ahead 1036；存在与本任务无关的既有修改和未跟踪artifact，本任务不覆盖或清理。
- 本报告同时镜像到根目录规定报告面：`E:\type10-7\automation_reports\CV-SincNet\qknnv42_lightweight_head_20260714_173701\report.md`。

## 7.首轮本地压缩结果

在原无FFT特征缓存上，以`receiver×seed×K=5×5×5=125`个同切分run重放三种head：

|head|old_acc|seen_new_acc|H_old_new|平均head MAC|dense graph下界|平均延迟/query|相对dense延迟|
|---|---:|---:|---:|---:|---:|---:|---:|
|原始dense LP|65.5933%|47.9400%|53.2562%|13.373 M|829,440 B|0.04230 ms|100%|
|support-prototype残差|66.3289%|44.7933%|51.1005%|1.966 M|0 B|0.02159 ms|51.05%|
|关闭LP，原old bias=+0.001|66.3867%|44.8667%|51.1787%|1.761 M|0 B|0.02133 ms|50.35%|

原bias下关闭LP的`seen_new_acc`下降3.073pp，超过3pp门槛0.073pp，因此不能直接晋升。随后只在seed 713101-713105诊断集上扫描old/new常数偏置，选定`old_anchor_bias=-0.001`；该扫描属于超参数选择，不作为最终验证。

使用完全未参与选择的seed 713106-713110重新生成125个确认run，结果如下：

|head|old_acc|seen_new_acc|H_old_new|Δold|Δnew|ΔH|平均head MAC|平均延迟/query|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|原始dense LP，bias=+0.001|64.6978%|47.3600%|52.6165%|0|0|0|13.373 M|0.04230 ms|确认基线|
|轻量无LP，bias=-0.001|64.4422%|47.8667%|52.9870%|-0.256pp|+0.507pp|+0.371pp|1.761 M|0.02061 ms|三个均值指标均通过≤3pp门槛|

确认集上，轻量head的估算MAC下降86.83%，dense query-query graph从至少829,440 B降为0，实测head延迟下降51.28%。该路径不依赖query batch、不使用query角色或类别配额，允许逐样本流式推理。125行中仍有9行old、26行new和19行H的单run降幅超过3pp，因此当前“≤3pp”只在预注册的矩阵均值口径成立；若要求每个receiver/seed/K行都满足，需要下一轮增加无标签置信度fallback。

本地artifact：

- 诊断：`E:\type10-7\automation_reports\CV-SincNet\qknnv42_lightweight_head_20260714_173701\artifacts\labelprop_125`
- 独立确认：`E:\type10-7\automation_reports\CV-SincNet\qknnv42_lightweight_head_20260714_173701\artifacts\labelprop_confirmation_seed713106_713110`

## 8.实现、验证与N607确认计划

|文件|改动|
|---|---|
|`paper_reproduction/cvs_aligned/cvs_method_runner.py`|新增`dense_transductive/support_prototype/disabled`三种LP模式、可配置old bias、query-query依赖标记以及MAC/内存元数据|
|`paper_reproduction/scripts/benchmark_qknnv42_labelprop_compression.py`|在相同缓存和split上运行125行配对资源/性能验证，支持独立seed grid|
|`tests/test_cvs_proposed_stage2_runner.py`|验证轻量模式不建立dense graph、不声明transductive query依赖且保持artifact契约|

本地验证：

|命令|结果|
|---|---|
|`conda run -n ssr-gpu python -m py_compile ...`|PASS|
|`conda run -n ssr-gpu python -m pytest -p no:cacheprovider -q tests/test_cvs_proposed_stage2_runner.py`|PASS，5 tests|
|本地125-run诊断+125-run独立确认|全部完成，无失败|

待同步文件及SHA256：

- `paper_reproduction/cvs_aligned/cvs_method_runner.py`→N607同相对路径，最终`2CCEB9D5E92DA4D520514632346FAB710BE30127760652048FBA4523EF21ED4F`。
- `paper_reproduction/scripts/benchmark_qknnv42_labelprop_compression.py`→N607同相对路径，`CAA482718D57A4C1E53323C00601007654A0A15EE89626E7057CEAFEA14B7ED3`。

N607只运行冻结FFT96特征上的CPU head确认，不启动训练、不占GPU。命令为：

```text
PYTHONPATH=<ROOT>/code:<ROOT> <PYTHON> -m paper_reproduction.scripts.benchmark_qknnv42_labelprop_compression --baseline-root <ROOT>/runs/cvs_qknnv42_fft96_singleview_125_20260714 --feature-cache <ROOT>/runs/cvs_publication_adv3b02_fft96_singleview_20260714 --out-root <ROOT>/runs/cvs_qknnv42_fft96_lighthead_confirmation_20260714 --modes dense_transductive disabled --seed-grid 713106 713107 713108 713109 713110 --light-old-anchor-bias -0.001
```

预期输出：125行dense基线+125行轻量结果、`paired_runs.csv`、`summary.json`及每行完整artifact。成功条件仍是独立确认矩阵的`old_acc`、`seen_new_acc`、`H_old_new`均值相对dense下降不超过3pp。

## 9.N607 FFT96独立确认结果

N607使用既有单视图FFT96缓存，在全新seed 713106-713110上完成125个dense基线和125个轻量run，`250/250 metrics.json`齐全，无失败。该确认没有重新训练backbone或adapter，也没有使用query标签选择bias。

|head|old_acc|seen_new_acc|H_old_new|Δold|Δnew|ΔH|平均head MAC|dense graph下界|平均延迟/query|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|FFT96+dense LP，bias=+0.001|74.4133%|65.2133%|68.6486%|0|0|0|22.725 M|829,440 B峰值；1,658,880 B累计|0.06582 ms|
|FFT96+无LP，bias=-0.001|74.3711%|65.7867%|68.9109%|-0.042pp|+0.573pp|+0.262pp|2.818 M|0 B|0.02353 ms|

相对同缓存dense基线，轻量head的估算MAC下降87.60%，dense query-query graph完全移除，修正口径后的正式预测延迟下降64.25%。两个分支的完整持久状态相同，均值为36,617.2 B（35.76 KiB），按K变化范围为22,678–62,806 B；该数字包含int8 support、类别索引/标签表、float64原型和Fisher center/scale，不再把raw support误报为0后遗漏其他状态。三个矩阵均值指标均未下降3pp，且`seen_new_acc`和`H_old_new`略升。逐run约束更严格：125行中old没有一行下降超过3pp，new有7行、H有2行下降超过3pp；因此当前版本满足“总体性能下降≤3pp”，尚未满足“每个receiver/seed/K行都下降≤3pp”。

本地结果：

- `E:\type10-7\automation_reports\CV-SincNet\qknnv42_lightweight_head_20260714_173701\artifacts\fft96_confirmation_seed713106_713110\summary.json`
- `E:\type10-7\automation_reports\CV-SincNet\qknnv42_lightweight_head_20260714_173701\artifacts\fft96_confirmation_seed713106_713110\paired_runs.csv`
- 复核修正版：`E:\type10-7\automation_reports\CV-SincNet\qknnv42_lightweight_head_20260714_173701\artifacts\fft96_confirmation_v2_truefft_seed713106_713110`

N607结果：

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_qknnv42_fft96_lighthead_confirmation_20260714`
- log：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/logs/cvs_qknnv42_fft96_lighthead_confirmation_20260714/benchmark.out`
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：未使用；CPU frozen-feature head重放。
- 远端SHA256与本地一致；远端`py_compile`通过。

## 10.5-view完整栈与资源优先级复核

N607上的完整分支也已完成125/125：60 epoch`id_norm_late_feature`+5-view TTA+FFT96+角色/类别配额Hungarian Oracle得到`old_acc=82.9289%`、`seen_new_acc=93.3733%`、`H=87.6485%`；head计时为`0.67059 ms/query`。同一125任务的单视图FFT96逐样本argmax为`75.1244%/64.6400%/68.5647%`和`0.16929 ms/query`。

这两个分支同时改变了adapter、TTA和决策Oracle，不能把19.084pp的H差距归因于5-view，也不能把完整分支当作可部署基线。它证明一件事：一次性删除`adapter+5-view+Oracle`虽显著省算，但性能损失远超3pp，路线不可晋升。角色/类别配额Oracle还使用普通星上推理不可获得的query角色与每类query数量，必须从部署方案中剔除。

首轮优化因此选择了可单独归因、可流式部署的dense LP：在保持单视图FFT96、相同特征、相同split和相同逐样本决策的条件下，head计算下降87.60%，主性能不降。下一轮针对端到端最大项5-view，应保留同一个adapter和逐样本决策，只比较1/2/3/5-view或置信度触发TTA；否则无法判断性能收益究竟来自TTA、adapter还是Oracle。

## 11.需求追踪终态

|ID|状态|验证|
|---|---|---|
|QK-L01|`verified`|完成整体与head模块排序；得到本地/N607 MAC、内存和延迟证据|
|QK-L02|`verified`|轻量无LP路径可配置、可流式、默认路径向后兼容；4项测试通过|
|QK-L03|`verified`|FFT96独立125-run：Δold=-0.042pp、Δnew=+0.573pp、ΔH=+0.262pp|
|QK-L04|`verified`|125个receiver/seed/K行均使用3个`leo_*_weak`场景，support/query互斥|
|QK-L05|`verified`|轻量manifest记录`query_used_for_transductive_inference=false`、无角色Oracle、无quota|
|QK-L06|`verified`|远端保留250个完整run；本地保存250行同row资源/性能CSV与summary|

首轮head压缩已完成严格实现与矩阵均值门槛验证。整个“qKNN端到端星上轻型化”目标仍未结束：5-view是最大端到端算力项，下一轮必须在控制adapter和决策模式后继续压缩，并再次执行≤3pp门槛。

## 12.代码复核修正与FFT96重验计划

独立代码复核发现首版资源元数据和benchmark约束有四处需要收紧：轻量模式仍可被配置为legacy角色/类别配额Oracle；持久化状态没有计入Fisher变换参数、类别索引和原型；FFT双分支的dense graph按累计分配量而非顺序执行峰值上报；延迟把`old_acc_before_increment`诊断调用与正式部署预测相加。上述问题不改变既有预测值和MAC公式，但会影响部署边界、内存与延迟口径。

本地已完成以下修正：

- `support_prototype/disabled`强制`per_sample_argmax`，benchmark也覆盖并断言该决策模式。
- 将support enrollment与query scoring拆分；support被转换并量化后即可丢弃raw support，持久状态显式统计int8 support code、类别索引、类别标签表、float64原型以及Fisher center/scale。
- FFT双分支的`dense_graph_bytes_lower_bound`改为顺序执行峰值`max(primary,aux)`，另以`dense_graph_cumulative_bytes`记录累计分配量。
- `adaptation_latency_sec`仅计正式部署预测；旧类增量前诊断另记`old_before_increment_diagnostic_latency_sec`。
- dense默认路径恢复原`adaptation_objective=qknnv42_int8_top1_proto45_old_anchor_labelprop`，保持artifact兼容。

本地验证：`py_compile`通过，`tests/test_cvs_proposed_stage2_runner.py`为`5 passed`。待同步SHA256为：

- runner：`7B2C0004EAF9467F8B706B04003F0E44A7D732D221A9CB877E3F618FEF963A62`。
- benchmark：`69658B6326E74F5C85C426F438ECEAF92CA56C0A74D5D899BC89884F7767CEC9`。

N607只读预检发现8个RIEI训练进程正在运行。按项目规则切换为monitor-only，不同步代码、不启动远端重验。为不阻塞资源口径修正，已从N607只读复制原FFT96特征缓存和125个baseline配置到本地，在`ssr-gpu`环境完成125行dense+125行disabled复核。预测指标与首版完全一致；正式预测延迟修正为`0.06582→0.02353 ms/query`，graph峰值修正为`829,440→0 B`，持久状态均值为35.76 KiB。远端v2目录保持不存在，没有覆盖既有artifact。

## 13.第二轮端到端最大项：TTA压缩实现

5-view需要每个物理样本执行5次ADV3B02+adapter前向和5次FFT sketch，是当前端到端最大星上算力项。为避免重新训练adapter造成混杂，已实现“一次60 epoch训练、并列导出多种TTA策略”的固定adapter实验：

该变化涉及satellite/LEO接收侧视图口径，因此在任何N607同步或启动前，已在`E:\type10-7\项目.md`补充TTA协议，并镜像到Git承载面的`docs/source_controls/PROJECT_PROTOCOL.full.md`和`docs/PROJECT_PROTOCOL.md`：TTA只允许在同一物理LEO观测、同一seed、同一split、同一checkpoint和同一adapter下比较，不得把不同LEO随机扰动或不同adapter的差异归因于view数量。

|策略|视图|每样本backbone前向|每样本FFT|相对5-view前端计算|设计目的|
|---|---:|---:|---:|---:|---|
|`none`|1|1|1|20%|最大压缩，验证单视图是否在同adapter上通过≤3pp|
|`rx_shift3`|3|3|3|60%|保留±2 sample时移稳健性，前端计算下降40%|
|`rx_cfo3`|3|3|3|60%|保留±1e-4残余CFO稳健性，前端计算下降40%|
|`rx_light5`|5|5|5|100%|严格同adapter性能基线|

实现文件：

- `code/export_spaceborne_features.py`：新增`rx_shift3/rx_cfo3`并统一TTA策略清单和view count。
- `code/scripts/train_apply_phase1_iq_preadapter_20260703.py`：新增`--export_tta_policies`，同一adapter依次导出1/3/3/5-view，不重复60 epoch训练。
- `paper_reproduction/scripts/benchmark_qknnv42_tta_policies.py`：固定`FFT96+disabled LP+per_sample_argmax+bias=-0.001`，对500个policy/receiver/seed/K行做5-view配对门槛验证。
- `paper_reproduction/scripts/run_cvs_qknnv42_tta_ablation_20260714.sh`：精确复用历史60 epoch adapter超参数，先导出四套同样本特征，再运行500行轻量head矩阵。

本地组合验证为`11 passed`，其中TTA/多策略导出测试文件为`6 passed`；两个Python CLI的`--help`、`py_compile`与Bash`-n`均通过。benchmark还会对每个run计算support/query split manifest的SHA256并与5-view基线逐行核对，输出根非空时拒绝覆盖。独立只读代码复核确认同一adapter、相同LEO role seed、500行矩阵和Oracle禁用均正确；复核指出自定义subdir模板可能把多个policy渲染到同一路径，现已在训练前强制检查所有policy输出目录唯一，避免完成60 epoch后才发现覆盖风险。由于N607当前有8个训练进程，本轮没有同步或启动；待GPU/训练lane空闲后，按报告中的固定脚本执行。晋升规则是先选视图最少且`old_acc/seen_new_acc/H_old_new`三个矩阵均值相对`rx_light5`下降均不超过3pp的策略；若1-view失败而3-view通过，则采用3-view，直接获得40%前端算力压缩。

待后续lane空闲时同步的SHA256：

- `code/export_spaceborne_features.py`：`E853732BA97F524E110C0890CE2D175AECDBE04A5B43C4DA8A291D1361D5036D`。
- `code/scripts/train_apply_phase1_iq_preadapter_20260703.py`：`3E65431A121833D7C550CA6DD89AC876104F124FBF2E41A0056257D044487AA3`。
- `paper_reproduction/scripts/benchmark_qknnv42_tta_policies.py`：`70DDFD9D9B017C9C7BAB880C779C40101734BDF19E3475D4411E28FE2CFEE529`。
- `paper_reproduction/scripts/run_cvs_qknnv42_tta_ablation_20260714.sh`：`2052E25507BCF78E7272DAD737BACF1BC6EB9B68D6031CF2F5775FA0E5EA3DE6`。

## 14.第三轮head压缩：support-memory从全K样本压缩为每类2个多样代表

N607在2026-07-14 18:20仍有8个RIEI GPU训练进程，因此继续保持monitor-only，不同步、不启动、不干预。等待期间，本地继续处理qKNN head中仅次于dense LP的`Q×S×D`support-memory top-1相似度开销。

新增`qknnv42_support_representation`：

- `all_support`：原始做法，保留每类全部K条int8 support。
- `class_medoid`：每类只保留最接近prototype的一条support。
- `class_diverse2/class_diverse4`：先选类medoid，再以余弦距离farthest-first补到每类2/4条，兼顾中心与类内多样性。
- `prototype_only`：support enrollment后只保留类prototype，不保留support code。

Fisher/whitening和prototype仍由全部K-shot support拟合；代表选择完成后，raw support和未入选int8 code均可丢弃。dense label propagation只允许`all_support`，压缩表示强制走`disabled LP+per_sample_argmax`部署路径。

在seed713101-713105诊断集上，`class_medoid`和`prototype_only`未通过；`class_diverse2`相对原始dense的`old/new/H`变化为`-1.909/-2.227/-2.120pp`，`class_diverse4`为`-0.731/-1.053/-0.882pp`，二者均通过。按照“满足≤3pp后选择计算最轻方案”的预定规则，选择`class_diverse2`，再使用完全未参与选择的seed713111-713115做125-run确认。

|确认集head|old_acc|seen_new_acc|H_old_new|Δold|Δnew|ΔH|head MAC|持久状态|平均support code数（主分支+FFT）|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|原始FFT96+dense LP+all support|75.5000%|65.8333%|69.5704%|0|0|0|22.725 M|36,617.2 B|121.6|
|FFT96+无LP+每类2个多样support|73.2844%|63.1933%|67.1001%|-2.216pp|-2.640pp|-2.470pp|0.918 M|24,367.6 B|28.8|

确认集三个矩阵均值指标均满足下降不超过3pp。资源变化：

- head MAC相对原始dense下降95.96%；相对上一版无LP全support轻量head继续下降67.44%。
- support code存储从15,564.8 B降至3,686.4 B，下降76.32%；完整持久状态下降33.45%。
- dense query graph仍为0。
- 将support enrollment与重复query scoring分开计时后，纯星上scoring从`0.03286`降至`0.00552 ms/query`，下降83.21%；包含一次enrollment摊销的整次预测从`0.04849`降至`0.02209 ms/query`，下降54.45%。
- enrollment由`2.435`增至`2.601 ms/support set`，增加6.83%，原因是farthest-first代表选择；该步骤只在注册或support更新时执行，不随每条query重复。

严格逐run口径仍未全部满足：125行中old/new/H分别有44/54/55行相对dense下降超过3pp。因此晋升结论仍限定为用户目标采用的矩阵均值≤3pp，不得写成所有receiver/seed/K行的最坏情况保证。

本地artifact：`E:\type10-7\automation_reports\CV-SincNet\qknnv42_lightweight_head_20260714_173701\artifacts\fft96_support_diverse2_confirmation_v2_seed713111_713115`。实现新增独立benchmark`paper_reproduction/scripts/benchmark_qknnv42_support_compression.py`，并将测试扩展为`9 passed`。TTA端到端500行矩阵仍是整个目标的剩余验证项。

## 15.N607最新占用边界

2026-07-14 18:40再次执行只读训练清单。N607上仍有8个RIEI训练进程，另有一个已由外部流程启动的`run_cvs_qknnv42_full_legacy_oracle_125_20260714.sh`任务，正在GPU3训练60 epoch、5-view、FFT96的历史strict legacy adapter，输出根为`runs/cvs_qknnv42_full_adapter5_fft96_strict_20260714_183556`。该任务不是本轮support-memory压缩或固定adapter TTA矩阵，不能作为1/3/5-view轻量化结论。按活动任务monitor-only规则，本轮未同步、未启动、未修改或干预任何远端任务；本地SSH客户端和到N607/桥接机的TCP22连接均已归零。
