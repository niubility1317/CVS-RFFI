# Phase1 ADV3B02+FastTrust-EFF SRC5八种子扫描报告

## 最小预登记

- Run ID：`phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1`
- 当前状态：`RUNNING`；正式16行扫描已于2026-08-28 21:17:54 CST从hotfix release启动并完成一次PID/CWD/cmdline/GPU/log增长绑定检查。
- 研究目的：在`SRC5_MAXP2`源域上比较同种子的ADV3B02从头训练对照与FastTrust有效子集，使用源域`V_select`冻结星地信道性能最好的FastTrust种子。
- 数据协议：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；四个角色均仅来自源接收机，物理样本两两不交；目标接收机数据不参与训练、校准、选模、候选重排或选择性重跑。
- 源域profile：`SRC5_MAXP2`；源接收机`1-19,18-2,19-2,2-19,3-19`，ManySig索引`1,3,4,6,8`；目标接收机`1-1,14-7,2-1,20-1,7-14,7-7,8-8`，ManySig索引`0,2,5,7,9,10,11`；源/目标接收机严格不相交。16行扫描只构建源接收机索引`1,3,4,6,8`的4日数据及源域`V_select`，不构建或访问目标接收机加载器；目标4日数据只属于冻结后的独立确认阶段。
- 初始化：两臂均从头训练。旧`ADV3B02_CORE90_SOFT_E200`checkpoint含`1-1,14-7,2-1`源数据，而这3个接收机在新profile中属于目标域，因此禁止作为本轮初始化。
- 矩阵：8个种子`713101`至`713108`；每张GPU绑定一个种子，并并发运行`ADV3B02_CONTROL`和`ADV3B02_FASTTRUST_EFF`，共16行、每卡两行。
- FastTrust-EFF：保留高可靠三头一致硬伪标签、类平衡上限、源域先验对齐和高可靠U卫星身份CE；取消固定50%身份回填、soft/candidate身份路由、时序稳定门、prototype证据、U prototype更新、cross-RX损失和nuisance损失。
- 训练预算：E200、U batch256、eval batch512、fused student forward；源域重评E1起每10epoch一次，最后20epoch每epoch一次；Sinc前端保持FP32数值路径。
- 选种规则：只读取每行最终checkpoint的源域`V_select`。主分数为`source_val_sat_hmean=H(clean,min(leo_clear_weak,leo_low_elev_weak,leo_rain_weak))`。FastTrust种子只有在相对同种子对照满足`LEO mean`提升、`LEO floor`不下降且clean下降不超过0.5pp时才可进入一次性目标确认；多个通过种子按主分数、LEO mean、LEO floor、clean、种子升序依次打破并列。若无种子通过，则记录负screen，不声明FastTrust提升，也不使用目标域反向重排。
- 目标确认数据：只在源域冻结最佳种子后核对`protocol_schema=p2_min_v1`、`phase2_data_status=VALIDATED_ONCE`、`capsule_id=536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2`、`split_id=260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25`；预测先形成artifact，独立scorer随后连接truth。目标结果不反馈选种、调参、重训或重跑。
- 本地环境/CWD：`ssr-gpu`；`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\phase2-canonical-union-maxq`。
- N607环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；进程实际CWD为`/home/szu2070436088`，所有代码、输入、输出和`PYTHONPATH`均以绝对路径绑定`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1_7ed99037_hotfix1`。
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1`；禁止覆盖已有路径。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1/dispatcher.log`、run root下`dispatcher_logs/<candidate>.log`和每行`train.log`。
- 预期artifact：每行`final_ssdg.pth`、`config.json`、`status.txt`、`metrics_joint.json`，以及clean和3个LEO弱场景各自的metrics/log。
- 技术停止规则：仅协议/query越权、错误seed/split/receiver/day、错误checkout、输出碰撞、无法启动、无prediction闭合、同一确定性训练前异常至少出现在两行、进程归属不清或可能影响无关任务时停止；性能低不停止。
- 代码/配置Git提交：`16736ec34edfd4b77924d0bfe9030a80d3161a1a`；分支`codex/phase1-fasttrust-eff-src5-20260828`，提交后本地`HEAD`与远端OID一致。
- `REJECTED_EXTRA_GATE`：单独设计审批、spec SHA、成员级hash、额外seal/receipt和重复数据验证不属于排他最小实验工作流，本轮不执行。

## N607发布与启动冻结

- 唯一release归档：本地`E:\type10-7\local_artifacts\releases\phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1_16736ec3.tar.gz`映射到远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1_16736ec3.tar.gz`；本地/远端唯一SHA256均为`215d4d2a82e7f5bb26807cbed26596ef108de23e80bb262e6722420d62f79453`，状态`VERIFIED`。
- base release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1_16736ec3`；远端Python编译与两个launcher的`bash -n`通过。
- hotfix release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1_7ed99037_hotfix1`；由base release复制后只同步`code/SSDG/train_ssdg.py`。修复提交`7ed990372e357a9ce7ebc705b41294f4f5f04416`已push并核对远端OID；该文件本地/远端SHA256均为`92daf77c0f8d954e81e0b1b35a98a568e614ebdb71cef1a071e2489a27eb209d`，远端编译与行号回读通过。
- 行为dry-run：`SEEDSCAN-ROW=16`、`from_scratch=16`、`FastTrust-EFF=8`、目标接收机参数不存在、正式run root未创建。
- N607只读预检：2026-08-28 20:54 CST，8张RTX3090均为0%利用率、1MiB显存占用，无计算进程；正式启动前再次核对并维持每卡最多两个训练进程。
- 首个smoke：`phase1_adv3b02_fasttrust_eff_src5_noquery_smoke_e1_20260828_r1`在模型/数据迭代前触发`MUSE schedule boundaries must be strictly increasing`，状态`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；该run root和日志完整保留，不覆盖、不作为性能结果。正式E200边界`1<17<41<69<161<181<201`合法，不受该E1 smoke配置错误影响。
- 第二个smoke Run ID：`phase1_adv3b02_control_src5_noquery_smoke_e1_20260828_r2`；GPU0、seed713101、M0、scratch、E1，只验证两臂共享的源接收机4日数据、`V_select`隔离、真实checkpoint和clean+三种LEO严格重建评估链路；FastTrust-EFF专属路由已由聚焦单测、远端dry-run和正式E200命令覆盖。
- 第二个smoke结果：首个训练batch后触发`UnboundLocalError: local variable 'rc4_route' referenced before assignment`，状态`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；run root和日志保留，无残留进程。根因是既有RC4遥测在M1–M3分支内才定义`rc4_route`，但M0日志字典无条件读取该局部变量。
- 定点修复：只在每个batch入口设置`rc4_route=None`，不改变M1–M3/RC4后续覆盖赋值与数值路径；新增顺序回归测试。相关107项测试、Python编译和`git diff --check`通过；唯一一次定点复审结论为原P0已修复且该两文件修复无剩余P0/P1。
- 修复后smoke Run ID：`phase1_adv3b02_control_src5_noquery_smoke_e1_20260828_r3`；沿用r2冻结的GPU0/seed713101/M0/scratch/E1源域配置，只更换为hotfix release和全新不可覆盖路径。
- smoke r3结果：`final_ssdg.pth`为epoch1，状态`ARTIFACTS_COMPLETE`；checkpoint回读`target_access=false`、唯一named loader为`source_v_select`、source split receipt中的target days/receivers均为空；strict reconstruction无fallback、missing、unexpected或shape mismatch。clean为57.422%，`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`分别为38.033%/37.128%/36.422%，每场景18,000条，15个receiver×LEO行完整；以上仅作链路smoke，不作性能结论。
- smoke精确命令：

```bash
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1_7ed99037_hotfix1 PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python CONTROL_PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ID=phase1_adv3b02_control_src5_noquery_smoke_e1_20260828_r3 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_control_src5_noquery_smoke_e1_20260828_r3 GPU=0 SEED=713101 INIT_MODE=scratch CANDIDATE_ID_OVERRIDE=SMOKE_CONTROL_SRC5_E1 MUSE_UNLABELED_BATCH_SIZE=256 EVAL_BATCH_SIZE=512 SOURCE_VAL_HEAVY_EVAL_START_EPOCH=1 SOURCE_VAL_HEAVY_EVAL_INTERVAL=1 SOURCE_VAL_HEAVY_EVAL_FINAL_WINDOW=1 SOURCE_VAL_HEAVY_EVAL_FINAL_INTERVAL=1 TOTAL_EPOCHS=1 LABEL_EPOCHS=1 PSEUDO_EPOCHS=0 WISIG_TRAIN_DAYS=0,1,2,3 WISIG_TEST_DAYS= WISIG_TRAIN_RXS=1,3,4,6,8 WISIG_TEST_RXS= WISIG_ALLOW_SHARED_DAYS_IF_RECEIVERS_DISJOINT=false PHASE1_SOURCE_ONLY_EVAL=true EVAL_ON=source_v_select EVAL_GROUP_LOADER=source_v_select ABLATION=NONE bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1_7ed99037_hotfix1/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh --only=M0 > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_control_src5_noquery_smoke_e1_20260828_r3/dispatcher.log 2>&1 < /dev/null &
```

- 正式精确命令：

```bash
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1_7ed99037_hotfix1 MATRIX=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1_7ed99037_hotfix1/configs/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828.json RUN_ID=phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1 PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python CONTROL_PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1_7ed99037_hotfix1/code/scripts/launch_phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1/dispatcher.log 2>&1 < /dev/null &
```

## 正式启动状态

- launcher PID：`1066256`；run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1`。
- 主训练进程：16个；GPU0–7均严格为2个进程，每卡同一个seed的`ADV3B02_CONTROL(M0)`与`ADV3B02_FASTTRUST_EFF(M3)`各1行。GPU/seed映射为`0/713101,1/713102,2/713103,3/713104,4/713105,5/713106,6/713107,7/713108`。
- 启动后GPU0–7利用率为89%–98%，显存占用约3.6–3.85GiB/卡；16个candidate目录和16个dispatcher日志均已形成，训练日志增长，无`Traceback`、`UnboundLocalError`、OOM或FastTrust零step系统错误。
- 当前只读监控；不因中间性能停止，不热补丁或重启健康进程。只有预登记系统技术失败才处理本run的精确进程树并保留partial artifact。

## 需求追溯

|ID|来源|要求|目标文件/产物|状态|验证|备注|
|---|---|---|---|---|---|---|
|`SRC5-ISO`|`项目.md`与新数据profile|源/目标接收机不相交，扫描阶段不得混入目标域|`code/dataset_wisig.py`、`code/SSDG/train_ssdg.py`|本地已实现|聚焦协议负测已通过；真实ManySig smoke待N607|默认旧跨日行为保持不变；扫描模式`target_access=false`|
|`FT-EFF`|历史FastTrust同row机制证据|只保留硬路由、类平衡cap、源先验和U卫星身份CE，禁止低可靠回填|`code/cvsrffi/muse_ssdg.py`、worker launcher|本地已实现|route单测已通过；远端launcher dry-run和真实checkpoint smoke待N607|不使用依赖污染旧checkpoint的RC4/anchor cache|
|`SEED8-PAIR`|用户当前请求|8个种子，每卡同种子对照与FastTrust各一行|matrix+dispatcher|本地已实现|矩阵结构测试已通过；N607 GPU/PID检查待执行|共16行|
|`SPEED`|用户当前请求与既有速度证据|U256、eval512、稀疏源域重评、fused forward|matrix+worker launcher|本地已实现|静态命令与聚焦测试已通过；运行日志计时待N607|不牺牲E200或最终四场景评估|
|`SOURCE-SELECT`|`项目.md`|只用`V_select`选种，target预测后独立评分|matrix+本报告+分析脚本|已冻结|配置回读+完整artifact分析|目标结果不得反馈研发|
|`NO-OVERWRITE`|`AGENTS.md`|不可覆盖run root并只停止本run进程树|dispatcher|本地已实现|远端dry-run+真实启动绑定检查待执行|不得使用广泛`pkill`|

## 本地实现与验证

- 代码实现：新增源域专用评估模式、显式4日接收机不相交保护、FastTrust硬伪标签不回填路由、scratch配置去除未使用checkpoint记录、16行双实验调度器及冻结矩阵。
- 聚焦验证：共享日期/接收机隔离负测、源域`V_select`唯一加载器、FastTrust硬路由、矩阵两行/GPU、协议测试、严格星地评估、速度配置和Sinc数值测试均通过；Python编译检查与`git diff --check`通过。
- 独立审查：唯一一次P0/P1审查覆盖训练器、FastTrust路由、数据构建器、worker、seed-scan launcher、矩阵、测试和两份报告，结论为未发现P0/P1问题；未增加白名单外gate。
- 本地Git Bash探针未获得`MSYSTEM=MINGW64`，因此未把WSL替代通道用于launcher验证；N607发布后执行远端`bash -n`、行为dry-run及真实checkpoint无query smoke。
- 根目录报告位于非Git目录；Git镜像为`automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1/report.md`，提交时使用精确强制stage以越过仓库对`automation_reports/`的默认忽略规则。

## 完整源域artifact分析与种子冻结

- 2026-08-29 18:56CST短连接只读复核：16/16候选均为`ARTIFACTS_COMPLETE`，所有主训练进程已退出，GPU0–7均空闲；每行`phase1_terminal_status.json=COMPLETE`、exit code为0、`final_ssdg.pth`存在。
- 完整性：逐行完整读取`train.log`、全部评估日志和dispatcher日志；每行`metrics_epoch.csv`与`metrics_epoch.jsonl`均恰为200行；clean及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`四个最终评估文件齐全。四场景均由epoch200checkpoint严格重建：`strict_requested/checkpoint_load_strict=true`，无fallback、missing、unexpected或shape mismatch，且`target_access=false`。
- 健康性：完整日志未见Traceback、CUDA OOM、UnboundLocalError、FASTTRUST_ZERO_STEP、RuntimeError或Killed。训练日志中关闭组件的`sat_cos=nan`、`p95=nandeg`和空测试`0/0`为诊断占位；逐单元扫描全部16份epoch CSV/JSONL与64份最终metrics JSON，未发现任何非有限数值。
- 预登记`V_select`gate逐seed结果：

|seed|FastTrust clean|FastTrust LEOmean|FastTrust LEOfloor|FastTrust hmean|相对同seed对照Δclean/ΔLEOmean/Δfloor(pp)|gate|
|---|---:|---:|---:|---:|---:|---|
|713101|98.5722|92.5741|91.2778|94.7849|+0.0444/+1.4741/+1.3611|PASS|
|713102|98.4667|93.1852|92.3556|95.3133|+0.0389/+1.5111/+1.6611|PASS|
|713103|98.5833|93.5130|92.9611|95.6897|+0.0389/+1.3500/+1.7000|PASS|
|713104|98.5389|94.0926|93.4389|95.9211|+0.0667/+0.9407/+1.3056|PASS|
|713105|98.6111|93.0648|92.2611|95.3305|-0.0167/+1.4519/+1.7500|PASS|
|713106|98.5611|93.2685|92.5056|95.4374|+0.0111/+1.2981/+1.5333|PASS|
|713107|98.4778|93.2667|92.1778|95.2237|+0.0333/+1.3093/+1.3444|PASS|
|713108|98.4944|93.1241|92.6000|95.4563|+0.0111/+1.3093/+1.5944|PASS|

- 16行最终源域`V_select`测试结果如下；每个数值均来自对应行epoch200最终checkpoint的独立严格重建评估：

|seed|方法|clean|leo_clear_weak|leo_low_elev_weak|leo_rain_weak|LEOmean|LEOfloor|主分数|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|713101|ADV3B02_CONTROL|98.5278|93.3444|89.9167|90.0389|91.1000|89.9167|94.0255|
|713101|ADV3B02_FASTTRUST_EFF|98.5722|94.4556|91.9889|91.2778|92.5741|91.2778|94.7849|
|713102|ADV3B02_CONTROL|98.4278|93.3000|91.0278|90.6944|91.6741|90.6944|94.4030|
|713102|ADV3B02_FASTTRUST_EFF|98.4667|94.4833|92.7167|92.3556|93.1852|92.3556|95.3133|
|713103|ADV3B02_CONTROL|98.5444|93.7222|91.2611|91.5056|92.1630|91.2611|94.7630|
|713103|ADV3B02_FASTTRUST_EFF|98.5833|94.5722|93.0056|92.9611|93.5130|92.9611|95.6897|
|713104|ADV3B02_CONTROL|98.4722|94.6056|92.1333|92.7167|93.1519|92.1333|95.1974|
|713104|ADV3B02_FASTTRUST_EFF|98.5389|95.3222|93.4389|93.5167|94.0926|93.4389|95.9211|
|713105|ADV3B02_CONTROL|98.6278|93.5611|90.7667|90.5111|91.6130|90.5111|94.3953|
|713105|ADV3B02_FASTTRUST_EFF|98.6111|94.6167|92.3167|92.2611|93.0648|92.2611|95.3305|
|713106|ADV3B02_CONTROL|98.5500|93.8444|90.9722|91.0944|91.9704|90.9722|94.6096|
|713106|ADV3B02_FASTTRUST_EFF|98.5611|94.7167|92.5056|92.5833|93.2685|92.5056|95.4374|
|713107|ADV3B02_CONTROL|98.4444|94.1389|90.8333|90.9000|91.9574|90.8333|94.4859|
|713107|ADV3B02_FASTTRUST_EFF|98.4778|94.8389|92.1778|92.7833|93.2667|92.1778|95.2237|
|713108|ADV3B02_CONTROL|98.4833|93.3222|91.1167|91.0056|91.8148|91.0056|94.5969|
|713108|ADV3B02_FASTTRUST_EFF|98.4944|94.1611|92.6111|92.6000|93.1241|92.6000|95.4563|

- 结论：8/8种子同时满足LEOmean提升、LEOfloor不下降和clean下降不超过0.5pp。按预登记排序规则冻结`S713104_ADV3B02_FASTTRUST_EFF`，不使用目标域信息参与排序。
- 冻结行的源域结果：clean`98.5389%`，`leo_clear_weak=95.3222%`，`leo_low_elev_weak=93.4389%`，`leo_rain_weak=93.5167%`，LEOmean`94.0926%`，LEOfloor`93.4389%`，主分数`95.9211%`。
- 同seed对照为clean`98.4722%`，clear`94.6056%`，low-elev`92.1333%`，rain`92.7167%`，LEOmean`93.1519%`，LEOfloor`92.1333%`，主分数`95.1974%`；FastTrust-EFF同row增量依次为`+0.0667/+0.7167/+1.3056/+0.8000pp`，主分数`+0.7238pp`。
- 本节仅是源域选种结论，尚未读取Phase2 query或truth。用户追加要求将一次性目标确认覆盖`SRC5_MAXP2`全部剩余目标接收机域：`1-1、14-7、2-1、20-1、7-14、7-7、8-8`。下一步仅对已冻结种子核对既有`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`，先完成全部7域的prediction-first预测artifact，再由独立scorer统一连接truth；各域结果不得反馈选种、调参、重训或选择性重跑。
