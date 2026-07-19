# D81地面干扰谱Cauchy中心：seed713102独立确认

## 实验登记

- 实验ID：`d81_ground_nuisance_cauchy_center_seed713102_20260720`
- 登记时间：2026-07-20 04:55 HKT
- 操作者：Codex
- 目标：在`rx20-1/K10/new5/3场景×5 outer folds`上，用独立确认种子`713102`复验D81；判断seed713101的联合改善是否具有种子稳定性。
- 方法锁：`ground_nuisance_cauchy_center`；代码提交`2f6a26d3`，首种子完成报告提交`930481d5`。
- 对照：同一seed、同一capsule上的D62 target-support-only基线；同时列出D81 seed713101同row结果。
- 假设：84个地面int8类中心只估计类中心化跨域干扰谱，并用固定一步Cauchy权重稳健估计target类中心，可降低support中心漂移噪声；地面原型不进入query评分，查询附加计算为0。

## 数据与协议

- `protocol_schema=p2_min_v1`；固定单次`LEO_weak`已接收IQ，support-only适配，query逐样本独立评分。
- receiver=`20-1`，seed=`713102`，K=`10`，实际每类/场景support=`8`，seen-new=`5`，scenes=`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`，outer folds=`0..4`。
- 数据状态：D18母缓存30/30 cells已`VALIDATED_ONCE`；本次复用`rx_20_1_seed_713102`，不重建、不重验、不改变物理ID、scenario assignment、support/query split或schema。
- 禁止项：无clean/source访问；无query truth、role Oracle、类配额、真实batch类数、global reassignment或query-dependent batch optimization。
- 地面组件：84个有效int8聚合单元；NPZ SHA256=`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`；manifest SHA256=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`；当前签名状态仍为`UNVERIFIED`，因此结果只能是开发诊断证据。

## 版本与本地验证

|项目|状态|
|---|---|
|Git承载面|`E:\type10-7\github_publish\CVS-RFFI-repo`，主分支存在大量不相关工作树改动，严格按路径提交|
|隔离worktree|`E:\type10-7\code\snapshots\d81wt`，登记前HEAD=`930481d5`|
|D81实现|`code/cvsrffi/stage2_d81_ground_nuisance_cauchy_center.py`，SHA256=`44111f8d7ecd0ffcfbd887c09468a167e4e1134bad3c2798bd7f0f5f89c3dc7a`|
|D81 runner|`code/scripts/probe_d81_ground_nuisance_cauchy_center.py`，SHA256=`85baac449d2cd1c5b21bff63ba9b01fe95bb2025fcdfa8ee3127ae41a5e99e82`|
|已完成验证|D81 unit+synthetic D62 stack 11/11 PASS；D62/D80/D81相邻链30/30 PASS；seed713101真实ground smoke PASS|

## N607封装与运行计划

- 远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/source`。
- 复用缓存：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713102`。
- 新输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_ground_nuisance_cauchy_center_seed713102_20260720`；不得覆盖D18或seed713101产物。
- 环境：N607`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；本地`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。
- GPU：authority/capsule构建为CPU；D81确认计划本地执行，不占N607 GPU。若改为N607执行，先再次记录GPU占用且不超过每GPU两条训练进程。
- 私钥：仅本机`authority_signing_private_ed25519.pem`使用；绝不上传N607。
- 过程：远端用D18精确source复核seed713102真实缓存并生成unsigned authority；本地Ed25519签名；远端生成authority bundle和exact-K row pair；本地生成path-free runtime authorization并执行D81。
- PID：短时前台封装命令，无长期PID；若任何步骤改用后台命令，将在本节补记PID、GPU和log。
- 计划log：本报告目录下`logs/*.stdout.log`、`logs/*.stderr.log`；远端封装log位于新输出根`logs/`。
- 预期输出：独立seed713102 authority/capsule/runtime authorization、105-row完整训练日志、30 target rows、性能summary、资源/量化/协议审计和`RECEIPT.json`。

## 成功、停止与升级条件

- 确认成功：seed713102对matched D62满足D81预注册联合门——B/A/N/H/F/J、三场景、逐类floor、mean-row floor和三类confusion均不回退，且A/H/F或old-to-new confusion至少一项严格改善。
- 失败：任何联合门回退、收益消失或只由不稳定单样本翻转构成，均记为确认失败并详细报告，不启动125矩阵。
- 协议失败：ground/query访问、hash drift、signature/seal、support/query不交或单观测约束任一失败即fail-closed。
- 升级边界：只有独立种子确认成功，且ground正式签名问题解决后，才考虑扩大到多receiver/多seed；当前不启动125实验。

## 初始远端证据

- 2026-07-20 04:48 HKT direct N607 preflight PASS：server time、project root和8张RTX3090可见。
- 8张GPU均约10MiB且无活动实验进程；seed713102三个场景NPZ与`cache_set.json`存在。
- 已确认seed713102没有现成capsule；本次生成独立authority，绝不复用seed713101的seed-specific签名。

## 完成结果

待实验完成后补充：完整同row性能表、逐场景/逐类/混淆、量化一致性、训练日志统计、资源、异常、缺陷、最终判定与下一实验。

## Capsule完成与确认入口修复

- N607 unsigned authority成功：lock SHA256=`7d23365047a6ca0d6da885170b6c8fb0056586cf940619cf6450e1f893b12365`；build receipt SHA256=`ff29c1e3c98bea752a6e20e8a406df1610b2553421f430af5a279f489694c01d`。
- 本地Ed25519 envelope SHA256=`09d97a8f70fcb40cc7540bc4941c9756ba961532fca366bf8d44a060aacf2327`；私钥未上传。
- N607 authority commit=`59e6a6e0afcd7b519829e8e1a6ccc25ae43d07acd05815013542ee4cf6cc1c79`；exact-K row pair的同场景support/query不交、跨场景物理ID不交和before/after旧类复用审计均PASS。
- before/after enrollment seal分别为`6c2961f6f29f74ec885a6279529c813c462209e8b2db4d8e743bcb9e5fb2754f`、`ee0050b28ed3f10f90c7a19677896cab02f1ceae50777884263643306d02c454`；runtime closure=`b0b7f2c2f87e66ecbeca99779688461e7161877271dd0195e0bcf2b95cb9606f`。
- 第一次D81确认启动在读取query前fail-closed：继承执行器只允许开发seed713101，错误为`D42 preregistered development cell must be receiver 20-1, seed 713101, K10, new5`；退出码1、无输出目录、无性能数据。
- 最小修复只在D81 probe增加显式`--d81-confirmation-seed`入口；允许集合固定为`713102..713106`，并同时核验before/after receiver、seed、K、旧类前缀和new5。未提供该参数时仍使用原开发单元锁，未修改D42通用执行器。
- 修复后D81 probe SHA256=`f18dd80dd2f38312dde41e07120f43dc9cd2b78afa82b54c65fe08236e513817`；专项测试13/13 PASS，相邻D62/D80/D81链32/32 PASS，`py_compile`与`git diff --check`PASS。
- 重跑命令在原锁定命令基础上只新增`--d81-confirmation-seed 713102`；其余capsule、ground、runtime和query边界不变。
- 第二次启动同样在query前fail-closed：seed713102使用新的匿名class handles，而命令仍传入seed713101的binding，触发`ADV3B02 class binding contract drift`；退出码1、无输出目录、无性能数据。
- 已按D19既有离线边界生成seed713102专属binding：`analysis/d81_seed713102_adv3b02_class_binding_20260720.json`，SHA256=`5d191dd02038c6568c9787819be5efc8067323496ef0c80c2ef825a147803b65`。6个handle严格等于before package的有序注册表；Phase1 TX、direct-logit行hash和int8地面列顺序保持不变。
- 用原D41 `_load_component`做真实组件加载验证通过：6类、25,428B逻辑状态；mapping来源仍为`offline_scorer_truth_sidecar_before_predictor_boundary`，`query_truth_exposed_to_predictor=false`。
- D81 seed713102完整运行成功：耗时123.27秒，105/105 training rows；receipt seed=`713102`、query未打开、source closure不变。完整性能解析前先补跑同capsule的D62 matched baseline。
- D62 probe同样增加显式`--d62-confirmation-seed`窄入口，验证规则与D81一致，未修改通用D42执行器；脚本SHA256=`39622a58e8b1b647577aa56ffd414f3efe50ed72bc821c2e15a590ecf9da694e`。D62/D80/D81相邻测试34/34 PASS，`py_compile`与`git diff --check`PASS。
- matched D62首次运行在48.5秒时因`D43 FP32 centering changed support argmax`fail-closed，未完成、不得用于性能比较。该保护比较未中心化FP32头与实际部署的中心化FP32头；中心化的FP64代数等价性仍成立。
- 为保留实际中心化部署头而不隐藏数值现象，只在D62确认入口显式允许继续，并新增每次fit的`d43_centered_support_fp32_argmax_equivalent`、changed count和drift-allowed审计；开发入口默认仍fail-closed。没有回退到另一预测头、没有标签选择或超参扫描。
- 修复后D43 helper SHA256=`0d7b3b493eeb9073236bb85bb9c71b4e71a9e3a691ab7055daed9ab2415766b3`，D62 probe SHA256=`833dac6035e03ecdfa1f170ec1a95b13365fcacb8677505a149a8ab90f77f9f8`；D43/D62/D80/D81链42/42 PASS，`py_compile`与`git diff --check`PASS。pytest退出后的Windows临时目录清理出现非实验性`WinError 5`提示，但测试命令退出码0且42项全部通过。
- matched D62 retry2已完整运行105行、耗时118.42秒；但D62 geometry摘要会裁剪新增的D43逐fit字段，无法从完成产物精确统计数值漂移次数。为满足完整缺陷报告要求，不修改预测结果，仅把D43逐fit等价/变化计数聚合写入D62 metadata后再做一次final retry。
- final retry代码：D43 helper SHA256=`f0115762f3d05106e5fe1c87df6e208ef9e44eec0458b7e4e477c04531d7379e`，D62 probe SHA256=`defe3829e0418fa5ea0577ca385197705901232c12145bf64abc504e140b74f4`；42/42 PASS，`py_compile`和`git diff --check`PASS。

## 最终完成判定

- 状态：`COMPLETED_CONFIRMATION_GATE_PASS_DEVELOPMENT_ONLY_UNVERIFIED_COMPONENT`。
- D81与matched D62均完成105/105 training rows；D81耗时123.27秒，D62 final retry耗时120.90秒。
- D81在seed713102再次满足预注册联合门：全部总体指标、三场景均值与下界、逐类下界、mean-row floor和混淆项均不回退，并在`A/H/F/J`及old→new confusion上严格改善。
- 但地面组件仍为`UNVERIFIED`，且绝对性能明显未达目标，因此本结果仅是开发诊断证据，不能升级为正式结果，也不启动125矩阵。

## 同row总体性能

|版本/seed|B旧类注册前|A旧类注册后|N新类|H_old_new|F遗忘|J旧类联合保持|min class B/A/N|mean-row floor B/A/N|old→new/new→old/new→new|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D62/713102|86.11%|71.67%|82.67%|75.5429%|14.44%|33.33%|76.67/50.00/66.67%|46.67/33.33/53.33%|33/13/13|
|D81/713102|86.11%|72.22%|82.67%|75.9302%|13.89%|36.67%|76.67/53.33/66.67%|46.67/36.67/53.33%|32/13/13|
|D81−D62|0.00pp|+0.56pp|0.00pp|+0.3873pp|−0.56pp|+3.33pp|0/+3.33/0pp|0/+3.33/0pp|−1/0/0|
|D81/713101|92.78%|82.78%|84.67%|82.9366%|10.00%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|22/8/15|

`B/A/N`分别为旧类注册前准确率、旧类注册后准确率和已见新类准确率；`F=B−A`。FP32与INT8所有15个outer rows逐样本预测完全一致，因此表中INT8指标同时等于FP32指标。

## 逐场景性能

|场景|版本|B|A|N|H|F|J|min class B/A/N|mean-row floor B/A/N|混淆old→new/new→old/new→new|
|---|---|---:|---:|---:|---:|---:|---:|---|---|
|clear|D62|91.67%|76.67%|84.00%|79.495%|15.00%|40.00%|80/70/70%|60/40/50%|9/5/3|
|clear|D81|91.67%|78.33%|84.00%|80.657%|13.33%|50.00%|80/70/70%|60/50/50%|8/5/3|
|low-elev|D62=D81|93.33%|80.00%|84.00%|81.218%|13.33%|50.00%|80/60/60%|70/50/60%|8/3/5|
|rain|D62=D81|73.33%|58.33%|80.00%|65.916%|15.00%|10.00%|50/20/60%|10/10/50%|16/5/5|

唯一改变发生在clear/fold0：D81相对D62的`A`+8.33pp、`H`+5.81pp、`F`−8.33pp、`J`+50pp、mean-row A floor+50pp，并挽回1个old→new错误；其余14/15 outer rows完全相同。seed713101的唯一收益发生在low-elev/fold0，说明方向可能跨seed有效，但覆盖率仍极低。

## 逐类性能

|类别|角色|B D62/D81|A或N D62|A或N D81|变化|
|---|---|---:|---:|---:|---:|
|14-10|旧类|80.00%|63.33%|63.33%|0|
|14-7|旧类|76.67%|50.00%|53.33%|+3.33pp|
|20-15|旧类|83.33%|76.67%|76.67%|0|
|20-19|旧类|86.67%|73.33%|73.33%|0|
|6-15|旧类|90.00%|70.00%|70.00%|0|
|8-20|旧类|100.00%|96.67%|96.67%|0|
|1-16|新类|—|96.67%|96.67%|0|
|1-18|新类|—|66.67%|66.67%|0|
|18-10|新类|—|83.33%|83.33%|0|
|14-11|新类|—|80.00%|80.00%|0|
|8-3|新类|—|86.67%|86.67%|0|

## 地面原型机制、训练、量化与资源

- 84个地面单元经类中心化后得到rank-14干扰子空间，保留79.7586%谱能量；无超参扫描。每类target support残差只执行一次Cauchy稳健加权，得到类公共的160维平移；新旧类使用同一公式，地面原型不按class ID进入评分。
- 归一化最小样本权重min/mean/max=`0.02056/0.02764/0.04682`；有效样本数min/mean/max=`6.906/7.321/7.997`；最终中心位移范数min/mean/max=`0.04272/0.05105/0.06280`。残差重构最大误差`2.78e-17`，FFT/RF分支变化为0。
- 20条训练trace、300条训练记录；loss min/mean/max=`0.0970/0.3282/1.1108`，support accuracy min/mean/max=`87.5/98.72/100%`。
- 15/15 rows的INT8相对FP32 argmax变化为0、margin flip为0；最大score error min/mean/max=`0.000411/0.000826/0.001643`。
- ground统计90,521,600 MAC，稳健平移21,890,560 MAC，D81新增总计112,412,160 MAC；总适配计算25,003,636,130 MAC。query附加计算为0，无dense query graph。
- 参数2,016、epochs=20、steps=20、持久状态34,011B（affine 8,583B+ground 25,428B）、ground basis瞬时18,032B、单query 6,624 MAC、峰值CUDA显存22,886,912B；全部资源上限通过。
- ground组件共fit 1,080次、transform 2,160次，NPZ/manifest hash全程不变。D62的FP32中心化数值审计：1,080次fit中4次非等价、每次最多改变1个support预测；final retry与retry2的15行外层预测hash和指标逐行一致，因此该数值现象未污染matched比较。

## 完整性与证据路径

- D81训练日志：`ground_nuisance_cauchy_center/training_log.jsonl`，18,367,480B，SHA256=`ff7dfce22b67c621263756a408bc8970c6402680fcb8656bde680e5b06c77950`。
- D81 metadata SHA256=`c10887aff08a424ca30aad747419ff32b5cdff7cd4b00896eeaad6c91e3bdaeb`；receipt SHA256=`0e7c80ec6f9076851a60cd2b72aeeff473e03ff8db2b6bdb14892a9b369f97cf`。
- D62 final训练日志：`matched_d62_crossfitted_fisher_row_splice_retry3/training_log.jsonl`，14,637,372B，SHA256=`f3c1147004f5be5b1f601b6bc319f0db7e311136c20d218a2acc9367e99f3fa0`。
- D62 metadata SHA256=`023a8274e706187d3c6d07346dca70dc77ace47dafee03bc8e5fe92faa2987ec`；receipt SHA256=`ad5e64e6db4c296090634687fb5471bf63c682ee8554b059f98d0c2d835a5d4a`。
- 完整15-row解析、逐类/逐场景、资源与联合门证据：`d81_seed713102_confirmation_summary.json`，96,127B，SHA256=`26fa5b06becfb4962e32ca53df2bd9f7e074e1f073036c9781d3a8569e36088c`。

## 缺陷、结论与下一轮

1. 两个seed的相对收益模式一致，但每个seed都只改变1个预测，方法覆盖率不足；它更像稀疏纠错器，尚未成为普遍域适应机制。
2. seed713102绝对性能比seed713101显著更差；距K10目标仍有`A −19.78pp`、`minA −34.67pp`、`new5 −9.33pp`。rain仍是主瓶颈：`A=58.33%`、最差旧类20%、旧→新错误16个。
3. D81只稳健估计类中心，没有利用已识别的地面干扰谱去收缩每个support样本的域内残差；因此无法系统改善rain和新类。
4. 下一轮D82固定为“D81稳健中心+地面干扰方向的参数无关Wiener式残差收缩”：从同一类中心化地面谱构造闭式收缩系数，对所有新旧类support使用同一变换，K1严格恒等，query不更新且不增加评分开销；禁止扫描收缩强度。
5. D82是D77-D79回顾后的第三个探索版本；完成后必须先做记录化回顾，复核新旧类、注册前后、逐类遗忘和协议边界，再决定D83，不能直接开启第四轮。
