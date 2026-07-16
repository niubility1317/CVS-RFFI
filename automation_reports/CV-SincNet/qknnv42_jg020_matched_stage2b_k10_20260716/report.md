# JG_R8_LR020严格配对Stage2-B K10实验报告

## 1.基本信息

|字段|值|
|---|---|
|实验ID|`qknnv42_jg020_matched_stage2b_k10_20260716`|
|时间|2026-07-16 11:55+08:00|
|操作者|Codex `/root`|
|当前状态|`COMPLETE_DIAGNOSTIC_NEGATIVE`|
|目标|在历史三域适应矩阵相同receiver、seed、K10物理support/query ID、三场景和support/query View种子公式下运行`JG_R8_LR020`|
|比较对象|MRIOR-SDA、DADDA-SDA、ProtoNet CDA、严格直接ADV3B02、P4 identity qKNN|
|阶段|Stage2-B target-old LEO_weak-only|

## 2.假设与方法

假设：`P4＋BPJG-LOPO joint_gate rank8,lr=0.02,5epoch,50step`在严格target K10上能够维持历史source筛选的正适应收益，同时以6400个可训练参数和一次缓存backbone前向显著降低MRIOR/DADDA的完整backbone训练计算。

固定方法：

|字段|值|
|---|---|
|基座|ADV3B02，checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|ground adapter|P4 projection_feature rank16，SHA256=`95f9a8bac7880d42f705db7f16523c37cf4ce5ff8438ac2c500c7550a38de446`|
|target更新层|`id_gate.0＋joint_proj.0`|
|target LoRA|rank8、alpha8、lr=0.02、weight decay=1e-4|
|训练预算|5epoch、最多50 optimizer step、SGD无momentum|
|可训练参数|6400|
|support View|同一K10物理ID的3个预注册`leo_*_weak`场景View；种子=`seed+1000+scenario_index`|
|query View|每个scenario固定1-view；种子=`seed+2000+scenario_index`；不做TTA|
|决策|全6个已注册旧类cosine prototype逐样本argmax|

## 3.严格配对矩阵

- target receiver：`20-1,3-19,7-14,7-7,8-8`。
- seed：`713101,713102,713103,713104,713105`。
- K：固定10个物理support/类；support pool上限20；query 20/类。
- 场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- 正式row：25个receiver×seed适配任务，每row同一adapter评估3个场景。
- ID/View门禁：JG密封包的K10 support/query ID必须与历史MRIOR/DADDA/ProtoNet对应split manifest顺序完全一致，同时绑定历史Git commit=`d7f2f549ceb4903c1ab8b219b44f581379deacf3`、runner SHA256=`1270dbdb40285393519796a65a4f9bce3a0a89debdfce0e9a3ca1521a930a9db`、`_apply_scenario`源码SHA256=`0441168c391db173db25501165098e0b7236d475003cfdb31b56f5a1f139a22d`及support/query种子公式；任一不一致即阻断Phase2启动。

## 4.权限与协议

- `phase2_sample_view_policy=leo_weak_only_no_clean_access`。
- clean sample、clean-derived signal、clean dataset/cache/control-flow均不可达。
- predictor不接收query truth、old/new角色、真实批次类别数量、类别quota、排序或全局分配信号。
- predictor先写不可变prediction artifact；独立scorer随后连接truth sidecar。
- 旧125矩阵不作为当前Phase2输入，只在Phase2边界外提供ID一致性审计和历史对照值。
- 本轮不含target-new，因此不声明Stage2-C、`seen_new_acc`或`H_old_new`。

## 5.本地版本状态

Git仓库：`E:\type10-7\github_publish\CVS-RFFI-repo`，分支`codex/cvs-rffi-release-20260626`。启动设计时分支相对远端ahead 1382。现有未提交修改包括JG020 NumPy/PyTorch兼容修复，以及用户已有的`mitigating_da_rootcause`计划文件；本轮不得覆盖后者。

根目录`E:\type10-7`不是Git仓库。本报告在根目录保存，同时镜像到Git承载面；代码、配置、脚本只在Git承载面修改并提交。

## 6.本地改动与验证

|类型|已完成|
|---|---|
|runner|严格ADV3B02 Stage2-B predictor增加锁定JG分支；主干冻结；缓存joint input后仅更新6400参数|
|View重建|Phase1专用导出器按历史K10顺序分别用`seed+1000+s`和`seed+2000+s`生成support/query View，只持久化LEO_weak IQ|
|计划生成|生成25个target cache、25个密封predictor bundle、25份runtime evidence和8个worker shard|
|矩阵worker|白名单允许且只运行`jg_r8_lr020`的25个K10 row|
|scorer|独立评分阶段同时计算direct ADV3B02、P4 identity qKNN和JG adapted qKNN|
|ID审计|逐receiver×seed严格顺序核对support/query ID、split字段、三场景、View种子和runner绑定|
|本地验证|`py_compile`通过；59个focused pytest通过；JG loss/episode与原实现数值等价；offline exact split selector通过；25-row plan dry-run通过；`git diff --check`通过|

本地验证命令与结果：

|验证|结果|
|---|---|
|相关脚本`python -m py_compile`|PASS|
|`pytest`：runner、协议、matrix、JG隔离、LEO cache、predictor bundle、旧plan|`59 passed`|
|新旧JG runtime primitive数值对比|`JG_RUNTIME_PRIMITIVES_PARITY_PASS`|
|exact split selector定向检查|`OFFLINE_EXACT_SPLIT_SELECTOR_PASS`|
|25-row计划与View种子审计|`JG_MATCHED_PLAN_25_ROW_DRYRUN_PASS`|
|`git diff --check`|PASS|

## 7.N607计划

|字段|计划值|
|---|---|
|远端工作目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|Conda环境|`CVS-RFFI`|
|run root|`runs/qknnv42_jg020_matched_stage2b_k10_20260716`|
|log root|`logs/qknnv42_jg020_matched_stage2b_k10_20260716`|
|GPU|GPU0–7各一个worker shard；每行仅6400参数适配，以启动前live inventory为准|
|启动命令|由`plan/plan_manifest.json`中的`phase2_workers`8条命令启动；先完成25 cache＋25 bundle＋ID/View audit＋runtime seal|
|预期输出|25份prediction/split/loss/metrics/detailed/score/access audit及总汇总|

## 8.启动前风险

1.现行严格target cache尚未构建，必须先完成25个Phase1离线cache和bundle。
2.N607为PyTorch2.1＋NumPy2.2；显式小规模复制兼容修复已在本地13项JG隔离测试内通过，仍需远端首行smoke确认。
3.JG使用三场景support-only增强，而历史MRIOR/DADDA按scenario分别适配；报告必须同时给出实际support前向和适配次数，不能把算法差异写成完全相同训练View预算。
4.只有旧类结果，不能代替仍在修复中的新类注册实验。

## 9.N607落地记录

截至2026-07-16 12:37+08:00，N607 direct preflight与live inventory均PASS，8张RTX3090空闲；本地文件已同步并逐文件核对SHA256。远端首单元`receiver=20-1,seed=713101,K=10`已完成Phase1 cache和predictor bundle。严格配对审计结果为`1/1 PASS`：60个support和120个query的物理ID集合及顺序均与历史MRIOR manifest一致，三场景顺序、1-view query、support/query View种子公式、历史runner/commit绑定与row身份合同全部PASS。

首轮远端准备曾在写cache前遇到LEO配置导入路径错误，第二轮在写cache前遇到N607 NumPy2.2/PyTorch2.1的`np.concatenate`ABI错误；分别由提交`502fadd`和`e82aa93`修复。审计工具随后修复receiver目录原样保留和audit展示字段，提交为`299c706`与`bc2283a`。这些失败均发生在正式prediction/metrics产生前，不构成实验结果。

计划启动的Phase1准备命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/run_adv3b02_three_da_cache_plan.py --plan-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/qknnv42_jg020_matched_stage2b_k10_20260716/plan/plan_manifest.json --execute
```

标准输出和错误写入`/home/szu2070436088/2510044040/CV-SincNet/logs/qknnv42_jg020_matched_stage2b_k10_20260716/cache_prep.log`，PID写入同目录`cache_prep.pid`。该阶段顺序补齐其余24个cache/bundle，随后执行25/25 ID/View硬审计并构建runtime seal；任一门禁失败都阻断Phase2 worker。

Phase1准备进程PID=`2348101`，于12:39启动并正常结束。最终生成25/25个cache、25/25个predictor package与detached seal；`matched_id_audit.json`为`25/25 PASS`，`same_support_query_physical_id_sets=true`、`same_query_view_strategy=true`；runtime seal生成6个顶层证据文件，cache prep完整日志错误扫描PASS。启动时GPU0另有一个CI矩阵子进程，约占298MiB；本轮Phase1为GPU0上的第二个短时任务，未干预现有进程且未超过单GPU两个训练进程上限。

计划执行的单row Stage2-B smoke命令如下；它对应worker shard0的首行，即`receiver=20-1,seed=713101,K=10`：

```text
cd /home/szu2070436088/2510044040/CV-SincNet && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/run_cvs_publication_matrix.py --phase stage2b --config /home/szu2070436088/2510044040/CV-SincNet/runs/qknnv42_jg020_matched_stage2b_k10_20260716/plan/phase2_config.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/qknnv42_jg020_matched_stage2b_k10_20260716/stage2_runs --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/qknnv42_jg020_matched_stage2b_k10_20260716 --methods jg_r8_lr020 --receivers 20-1,3-19,7-14,7-7,8-8 --k-grid 10 --seeds 713101,713102,713103,713104,713105 --module-override paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner --shard-count 8 --shard-index 0 --device cuda:0 --post-prediction-scorer paper_reproduction/scripts/score_adv3b02_three_da_predictions.py --scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/qknnv42_jg020_matched_stage2b_k10_20260716/phase2_scoring_sidecars --isolation-launcher code/scripts/run_phase2_landlock_isolated.py --runtime-allowlist /home/szu2070436088/2510044040/CV-SincNet/runs/qknnv42_jg020_matched_stage2b_k10_20260716/runtime_seal/artifact_member_allowlist.json --runtime-evidence-root /home/szu2070436088/2510044040/CV-SincNet/runs/qknnv42_jg020_matched_stage2b_k10_20260716/runtime_seal --isolation-runtime-read-dir /home/szu2070436088/.conda/envs/CVS-RFFI --max-rows 1 --execute
```

## 10.结果表

第一次Stage2-B smoke于12:42启动，PID=`2372367`，3.48秒后fail-closed退出，`completed=0,failed=1`，没有生成prediction、metrics或loss trace，因此不是性能结果。完整row日志显示唯一异常为`Phase2ContractError`：runner错误地按scenario分别使用`scenario/support_artifact/query_artifact`单数键，而合同要求一次性三场景列表`scenarios/support_artifacts/query_artifacts`。完整65个文件访问trace共28,510行、3,683,618字节；未出现ManySig、ManyTx、clean、truth sidecar、query role或class quota访问，失败发生在IQ训练前的请求schema验证阶段。

修复提交`3cd7bfd`将request构造抽为可测试函数，并使用三场景有序列表一次性验证；32项runner/request/runtime-contract相关测试PASS。失败smoke的35个日志与旧runtime seal文件已归档至`E:\type10-7\automation_reports\CV-SincNet\qknnv42_jg020_matched_stage2b_k10_20260716\evidence\failed_smoke_20260716_1242`，共192,487字节。同步修复runner后必须在原`runtime_seal`路径重建runtime代码哈希与25份evidence，再只重跑同一row；其余24行继续阻断。

修复runner`3cd7bfd`同步时local/remote SHA256均为`35a1ccaf579b7d0f2522c7d6e791d73839bffd3c3d07b32f546562582841cbcc`。随后发现cache-plan执行器会把已有runtime seal作为已验证项跳过，因此当时记录的`runtime_code_sha256=43fabb...`仍是旧seal，不应表述为已重建；该控制面问题没有产生性能结果。

第二次smoke于12:49启动，PID=`2420316`，3.78秒后在完成request合同验证后退出，仍未进入训练或生成metrics。异常为`KeyError: candidate_lock`：predictor package按当前正式schema只把candidate lock的SHA256写入密封manifest，并没有将其作为可读取member；runner却错误地尝试读取不存在的member。修复提交`cfa058a`改为直接验证密封manifest中的candidate lock digest等于预注册JG_R8_LR020 lock SHA256=`43b25b78ec04e77a8442ae9c7dfe587868f91baf62f73a4c01d32697c00bf2a9`，同时保留`_validate_config`对rank、lr、epoch、View等全部字段的逐项锁定。33项相关测试PASS。第二次失败日志与当时runtime seal已归档至`evidence\failed_smoke_20260716_1249`，35个文件、195,194字节。同步后再次重建runtime seal并只重跑同一row。

`cfa058a`runner已同步，local/remote SHA256=`16a85417542f6580c54722fee4a6ecf3c26b4afdf9e45b11ae8e3403ace58f7b`。本次直接执行plan中登记的`build_phase2_runtime_seal.py`完整命令而非cache-plan跳过路径，当前`runtime_code_sha256=e1910a68138a5e46660d75d0b86b01bc74cf59441a75fd72f51026b57e60e466`，已确认相对旧hash发生变化；25份evidence、preopen audit和直接reseal完整日志错误扫描均PASS。现在才满足第二次修复后的smoke重跑条件。

第三次同row smoke于12:52前完成，worker汇总为`completed=1,failed=0`，所有prediction、loss、scoring、filesystem/runtime audit和详细指标artifact齐全。`filesystem_access_audit.status=PASS`、`landlock_enforced=true`、`forbidden_access_hits=0`；独立scorer在predictor退出后才打开truth，360个query-scenario预测全部完成。

首行`20-1/713101/K10`严格配对结果如下：

|方法|old_acc|相对direct|适配时延/row|更新预算|
|---|---:|---:|---:|---:|
|MRIOR-SDA历史同row|91.6667%|+21.9444pp|17.3683s|600次ADV3B02梯度更新|
|DADDA-SDA历史同row|80.2778%|+10.5556pp|13.0822s|600次ADV3B02梯度更新|
|JG_R8_LR020本轮|78.8889%|+9.1667pp|1.3385s|50次小adapter step，0次ADV3B02梯度更新|
|P4 identity qKNN本轮|76.6667%|+6.9444pp|—|0|
|strict direct ADV3B02|69.7222%|0|0|0|
|ProtoNet CDA历史同row|66.1111%|-3.6111pp|0.0335s|0|

JG相对P4 identity qKNN为`+2.2222pp`，但相对MRIOR为`-12.7778pp`、相对DADDA为`-1.3889pp`。三场景old_acc分别为83.3333%、73.3333%、80.0000%；最低类分别为55%、20%、20%，跨三场景聚合最低旧类`20-19`仅31.6667%，明显未达到当前`old_acc>=88%`和`min_old_class_acc>=85%`目标。loss在5个epoch内由2.7648降至2.1254，support accuracy由57.7778%升至64.1667%，margin由0.1821升至0.2258，全部有限；说明训练健康，但当前单原型几何仍压缩难类边界。

## 11.三轮探索回顾（2026-07-16 12:52+08:00）

本回顾在第四轮动作前完成。已重新读取`项目.md`2026-07-15版，刷新conversation index至979条，并检索`JG_R8_LR020/K10/新类注册/MRIOR`与`adapt/old_acc/88/新类注册/qKNN`。重点复核历史线程`019f36ac-3ee7-7c33-a7f9-e4717d9d26b3`、`019f5f8d-d807-71a3-91fd-292f0747836c`、`019f60fd-7930-7751-87de-d6986c0e6c03`、`019f649b-1632-78d3-a6a1-c7aa8935879d`，以及历史三域适应报告、JG new5/10/20注册development报告和K1 support信赖适应报告。

三轮路径与教训：

1.第一轮在训练前被三场景request schema硬门拒绝，说明不能沿用逐scenario单数artifact接口；修复为正式三场景有序列表合同。
2.第二轮在训练前暴露candidate lock只以digest存在于密封manifest、不是可读取member；修复为校验预注册digest和逐字段锁定，而不是扩大package输入。
3.第三轮完整通过，证明LEO_weak-only、无query truth/role/quota/global assignment、密封prediction后独立评分和6400参数缓存JG链路已落地；同时首行性能明确低于MRIOR且最低类严重不足，不能因计算轻就宣称目标完成。

历史路线吸取的关键经验：

- 历史`88.8354%`来自source-old、不同切分、单seed诊断，不能替代当前target matched Stage2-B或Stage2-C证据。
- JG new5/10/20 development虽注册前旧类适配为正，但注册后old/new/H显著塌缩；因此当前old-only矩阵不能代替新类注册，更不能晋级为完整Phase2主线。
- K1仍需同candidate、同query和同View下证明相对strict direct至少+2pp且配对CI下界大于0；本轮K10结果不能外推到K1。
- 不再考虑query角色Oracle、类别配额、Hungarian/global assignment或clean派生补丁；这些不是可接受的性能修复路径。

下一步决策：继续启动剩余24个receiver×seed K10 row，因为这是用户明确要求的25行严格配对Stage2-B测量，单行不足以判断跨receiver/seed分布，且每行仅6400参数、5epoch/50step，计算开销有限。该矩阵保持`diagnostic/comparative Stage2-B`声明边界；不得使用25行query结果回调JG超参数，也不得据此声明显著优于MRIOR。完成后先做25行paired均值、CI、receiver/class floor和资源Pareto审计；随后主线候选仍必须在同一run同时报告注册前old和注册new5/10/20后的old、seen_new、H、最低类、遗忘，旧类适配与新类注册保持同等优先级。

协议复核：25/25 ID/View硬审计仍PASS；clean sample/derived/cache/control-flow不可达；query truth、old/new角色、真实批次数、类别quota和全局分配均不可达；预测先密封、scorer后连接truth。当前row无target-new，因此只允许Stage2-B结论。

## 12.全量结果表

三轮回顾后于2026-07-16 13:01:24+08:00启动8个固定worker shard，GPU0–7各一条。启动前N607 direct preflight和live inventory均PASS，8张GPU无活动训练进程。完整命令、PID和GPU绑定写入`full_launch_receipt.json`；PID为`2480936–2480943`。8个worker均正常退出，汇总为`completed=24,skipped=1,failed=0`；跳过项是已经完成的首行smoke，因此有效结果为25/25。

### 12.1总体与配对结论

配对95%CI按25个相同receiver×seed row的差值计算，使用双侧`t(24)`临界值2.0639。

|方法|25行old_acc均值|最差/最好row|JG相对方法的配对差值|95%CI|JG胜/平/负|
|---|---:|---:|---:|---:|---:|
|JG_R8_LR020|78.8222%|61.6667%/91.1111%|—|75.4939%–82.1505%|—|
|strict direct ADV3B02|75.2111%|59.7222%/91.1111%|**+3.6111pp**|**+1.9647–+5.2575pp**|20/1/4|
|P4 identity qKNN|77.6333%|59.4444%/91.3889%|**+1.1889pp**|**+0.6218–+1.7560pp**|20/0/5|
|MRIOR-SDA|84.5000%|69.7222%/93.6111%|**-5.6778pp**|**-7.6463–-3.7092pp**|3/0/22|
|DADDA-SDA|79.3556%|63.0556%/90.8333%|-0.5333pp|-1.5732–+0.5065pp|8/0/17|
|ProtoNet CDA|70.8556%|51.6667%/87.5000%|**+7.9667pp**|**+6.2628–+9.6705pp**|25/0/0|

结论边界：JG对strict direct和P4 identity的提升均为配对CI下界大于0，说明K10轻量梯度适配确实带来统计上稳定的正收益；但它对MRIOR显著落后5.6778pp，对DADDA均值落后0.5333pp且CI跨0。因此本版本满足“适配应优于直接ADV3B02”的方向性要求，但不满足`old_acc>=88%`，也不满足“显著优于MRIOR”的主目标。

### 12.2receiver与场景分解

|receiver|direct|P4 identity|JG|MRIOR|DADDA|ProtoNet|JG聚合类floor均值/最差|
|---|---:|---:|---:|---:|---:|---:|---:|
|20-1|66.3333%|72.6667%|75.2778%|85.5556%|75.7778%|62.2222%|44.3333%/30.0000%|
|3-19|62.7778%|64.4444%|66.1667%|72.7222%|66.2778%|56.2778%|44.0000%/30.0000%|
|7-14|89.3889%|89.0556%|89.0556%|87.9444%|88.8333%|85.8333%|71.3333%/63.3333%|
|7-7|83.0556%|82.5000%|83.0000%|90.2222%|83.6667%|76.0556%|56.3333%/38.3333%|
|8-8|74.5000%|79.5000%|80.6111%|86.0556%|82.2222%|73.8889%|59.3333%/51.6667%|

|query场景|direct|P4 identity|JG|JG-direct|JG-identity|
|---|---:|---:|---:|---:|---:|
|`leo_clear_weak`|77.1667%|80.5000%|81.4667%|+4.3000pp|+0.9667pp|
|`leo_low_elev_weak`|74.4333%|76.3667%|76.8667%|+2.4333pp|+0.5000pp|
|`leo_rain_weak`|74.0333%|76.0333%|78.1333%|+4.1000pp|+2.1000pp|

收益主要来自低基线receiver`20-1`、`3-19`和`8-8`；高基线receiver`7-14`上JG均值反而比direct低0.3333pp，`7-7`低0.0556pp。低仰角场景的identity→JG增益只有0.5000pp，表明当前JG更新对最困难View的额外对齐不足。

### 12.3旧类floor

每row先把同一旧类在3个场景的60个query合并，再取6个旧类中的最小值。25行该聚合floor均值为55.0667%，最差30.0000%，最好也只有76.6667%；场景×类cell的row floor均值43.2000%，全局最差15.0000%。因此25/25行均未达到`min_old_class_acc>=85%`。

|旧类|25×3个receiver-seed-scenario cell加权准确率|cell最差/最好|
|---|---:|---:|
|14-10|67.8667%|20.0000%/95.0000%|
|14-7|68.2000%|35.0000%/100.0000%|
|20-15|88.8000%|45.0000%/100.0000%|
|20-19|65.0000%|15.0000%/100.0000%|
|6-15|92.8000%|45.0000%/100.0000%|
|8-20|90.2667%|50.0000%/100.0000%|

均值由`6-15/8-20/20-15`三个强类支撑，而`14-10/14-7/20-19`长期低于70%。这说明当前损失下降并没有转化成worst-class边界保护，后续不应只优化mean old_acc。

### 12.4逐row同配对结果

`floor`为该row跨3场景聚合后的6类最低准确率。MRIOR、DADDA和ProtoNet均来自相同receiver、相同seed、K10历史row；support/query ID与View公式已由25/25硬审计确认一致。

|receiver|seed|direct|identity|JG|JG-direct|JG-identity|MRIOR|DADDA|ProtoNet|floor|JG时延|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|20-1|713101|69.72%|76.67%|78.89%|+9.17pp|+2.22pp|91.67%|80.28%|66.11%|31.67%|1.339s|
|20-1|713102|68.33%|76.67%|78.33%|+10.00pp|+1.67pp|88.06%|73.33%|67.78%|58.33%|1.210s|
|20-1|713103|60.28%|65.28%|67.78%|+7.50pp|+2.50pp|78.06%|71.11%|56.94%|48.33%|1.355s|
|20-1|713104|66.67%|70.56%|75.83%|+9.17pp|+5.28pp|89.72%|77.78%|58.33%|30.00%|1.895s|
|20-1|713105|66.67%|74.17%|75.56%|+8.89pp|+1.39pp|80.28%|76.39%|61.94%|53.33%|1.610s|
|3-19|713101|63.33%|70.00%|70.83%|+7.50pp|+0.83pp|75.00%|70.00%|58.89%|58.33%|1.464s|
|3-19|713102|59.72%|59.44%|61.67%|+1.94pp|+2.22pp|72.22%|64.17%|54.17%|36.67%|2.220s|
|3-19|713103|61.67%|63.06%|64.17%|+2.50pp|+1.11pp|73.61%|65.28%|56.94%|45.00%|2.194s|
|3-19|713104|64.44%|65.00%|67.22%|+2.78pp|+2.22pp|69.72%|68.89%|59.72%|30.00%|1.945s|
|3-19|713105|64.72%|64.72%|66.94%|+2.22pp|+2.22pp|73.06%|63.06%|51.67%|50.00%|1.955s|
|7-14|713101|89.17%|88.89%|89.72%|+0.56pp|+0.83pp|86.11%|89.72%|85.83%|73.33%|1.942s|
|7-14|713102|87.78%|86.94%|86.67%|-1.11pp|-0.28pp|90.00%|88.06%|83.06%|68.33%|1.747s|
|7-14|713103|91.11%|91.39%|91.11%|0.00pp|-0.28pp|85.83%|90.83%|87.50%|76.67%|1.745s|
|7-14|713104|90.00%|90.00%|90.28%|+0.28pp|+0.28pp|88.33%|87.78%|86.11%|75.00%|2.034s|
|7-14|713105|88.89%|88.06%|87.50%|-1.39pp|-0.56pp|89.44%|87.78%|86.67%|63.33%|1.372s|
|7-7|713101|83.33%|81.94%|84.44%|+1.11pp|+2.50pp|89.44%|85.00%|75.56%|56.67%|1.869s|
|7-7|713102|86.67%|87.50%|88.06%|+1.39pp|+0.56pp|93.61%|85.28%|81.94%|70.00%|2.064s|
|7-7|713103|79.44%|76.39%|76.67%|-2.78pp|+0.28pp|88.06%|82.22%|69.17%|38.33%|2.028s|
|7-7|713104|84.17%|85.00%|83.89%|-0.28pp|-1.11pp|90.56%|84.17%|80.56%|58.33%|1.961s|
|7-7|713105|81.67%|81.67%|81.94%|+0.28pp|+0.28pp|89.44%|81.67%|73.06%|58.33%|1.974s|
|8-8|713101|73.33%|79.72%|80.83%|+7.50pp|+1.11pp|88.33%|84.17%|74.17%|61.67%|2.026s|
|8-8|713102|75.28%|81.94%|84.17%|+8.89pp|+2.22pp|87.22%|86.67%|76.11%|70.00%|1.660s|
|8-8|713103|74.44%|77.78%|77.22%|+2.78pp|-0.56pp|83.61%|81.67%|70.28%|51.67%|1.902s|
|8-8|713104|75.00%|80.00%|81.94%|+6.94pp|+1.94pp|85.28%|79.44%|72.78%|58.33%|1.727s|
|8-8|713105|74.44%|78.06%|78.89%|+4.44pp|+0.83pp|85.83%|79.17%|76.11%|55.00%|1.812s|

### 12.5资源与训练健康度

|资源项|JG_R8_LR020|MRIOR|DADDA|ProtoNet|
|---|---:|---:|---:|---:|
|平均适配时延/row|1.8020s|16.7159s|13.1024s|0.0363s|
|相对JG时延|1×|9.28×|7.27×|0.020×|
|target optimizer step|50|600|600|0|
|ADV3B02梯度更新|0|600|600|0|
|target可训练参数|6400|完整方法训练路径|完整方法训练路径|0|
|峰值CUDA分配|23,843,328B，约22.74MiB|历史产物未记录同口径值|历史产物未记录同口径值|历史产物未记录同口径值|
|持久化增量状态估计|57,076B，约55.74KiB|未记录同口径值|未记录同口径值|prototype|
|support/query View|3/1|每scenario单独适配/1|每scenario单独适配/1|每scenario/1|

每row只执行180个样本等价的完整backbone前向，并把1980个support小子图前向用于5epoch/50step适配；query端合并LoRA后不增加额外adapter层。JG比MRIOR快约9.28倍、比DADDA快约7.27倍，且不做ADV3B02反向；但ProtoNet仍快约49.6倍。

25份loss trace共125个epoch记录，所有数值有限。平均loss由epoch1的2.1327降至epoch5的1.7693，平均support train accuracy由71.4333%升至73.5333%；没有NaN/Inf/OOM。训练收敛健康，但query类floor仍低，支持“优化目标与最差类泛化不匹配”而不是“训练崩溃”的判断。

### 12.6协议与artifact审计

|审计项|结果|
|---|---:|
|cache/package与历史ID-View硬审计|25/25 PASS|
|metrics/prediction/score/detailed/loss artifact|25/25完整|
|filesystem access audit|25/25 PASS，Landlock生效，forbidden hit=0|
|runtime preopen与sealed package|25/25 PASS|
|predictor先退出、scorer后打开truth|25/25 PASS|
|query role/true batch count/class quota/global assignment|25/25均未使用|
|完整成功日志扫描|33文件、233行、242,509B，错误命中0|
|worker汇总|24完成＋1跳过＋0失败|

远端证据归档为`runs/qknnv42_jg020_matched_stage2b_k10_20260716/completed_matrix_evidence.tar.gz`，SHA256=`6ddbaaf00066a1eab4643386f1d9f7a06aeb0b57efed8b1a6a1cf2e9991dd8e5`，包含424个索引文件、原始总字节9,061,069；本地副本和解包证据位于`evidence/completed_matrix_20260716_1301`，下载后SHA256一致。可复现汇总输出位于`analysis/matched_k10_full`，`summary.json`为`artifact_complete=true`且`errors=[]`。

### 12.7最终判定与下一步

本轮最终判定为`COMPLETE_DIAGNOSTIC_NEGATIVE`：

1.JG_R8_LR020在严格同receiver、support/query ID、K10、seed和View策略下对direct ADV3B02有显著正收益，证明“极轻量关键层适配”方向有效。
2.平均78.8222%与最低类30.0000%远低于旧类88%/最低类85%目标；并且显著低于MRIOR，不允许晋级为最强版本。
3.当前实验是Stage2-B old-only，没有新类注册、`seen_new_acc`、`H_old_new`或加入新类后的遗忘指标。不得把本轮结果解释为完整“域适应＋新类注册”性能。
4.下一轮算法重点应把`JG-R8`保留为轻量骨架，但将mean loss改为class×View worst-group保护并加入support-only信赖门；优先修复`14-10/14-7/20-19`三个弱类和低仰角View。任何候选都必须在同一run继续完成注册前old和注册new5/10/20后的old、seen-new、H、最低类与遗忘，且不能使用query结果调参。
