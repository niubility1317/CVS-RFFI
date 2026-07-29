# CVS-RFFI Phase2 T1 fresh v9运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2_t1_20260730_v9_d1f5e45c`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|复用v5的2个完整predictor package和v8通过的K=10 smoke，补齐其余48个package、feature cache、Stage2-A scoring sidecar、binding registry与plan seal，随后启动冻结states矩阵|
|Git实现|`d1f5e45c72f20e6d81ea5d6fef5e05fcd5f56f0e`|
|状态|`PREREGISTERED / INPUT_COMPLETION_PENDING / NO_PERFORMANCE_RESULT`|

## v8输入补齐批次关闭

v8的CPU formal预检和GPU smoke均通过。其后package补齐runner PID=`975269`启动48个缺失package，首批两个不同identity在prediction前出现相同确定性异常指纹：

```text
ValueError: current Phase1 class-label binding is unreadable
```

根因是批次路径替换将既有不可变class-label binding误指向不存在的v8 source路径。runner自然结束，48/48失败、0 seal、未进入feature/GPU阶段。原input、日志和summary保留，标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不覆盖、不续跑。

## v9复用与发布位置

|字段|值|
|---|---|
|N607 Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_t1_20260730_v8_d1f5e45c`|
|input|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_t1_20260730_v7_d1f5e45c`|
|controller|`<release>/artifacts/v9_input_completion/package_completion_controller.py`，位于input根之外|
|source summary|将Git承载面的v4 `package_build_summary.json`同步到`<release>/artifacts/v9_input_completion/`，位于input根之外|
|driver log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2_t1_20260730_v9_d1f5e45c`，独立fresh根|
|可复用package|v5 `rx20-1/method7283101/{before,new20}`，2/50|
|可复用feature|v8 smoke的Stage2-A/B/C K=10，3份|
|class-label binding|直接引用既有v2不可变文件，不做路径替换或重新验证|
|states plan|`stage2_states_plan_d1f5e45c.json`，SHA256=`f8d4687b05ffeada700d9e1b76da30b5fbc4e8657e50e7e2b08cf97a767d77f2`|
|并发限制|GPU0不使用；GPU1最多增加1个进程；GPU2-7各最多2个进程；每GPU总进程数≤2|

## 启动模板与停止规则

package补齐使用commit内`code/scripts/build_cvs_stage2_predictor_bundle.py`及Git承载的v4 50行命令矩阵，仅跳过v5已有2行；除脚本位置和三个输出flags外不改参数，class-label binding保持v2原路径。controller与source summary一起同步到release下独立`artifacts/v9_input_completion/`；controller启动时v7 input必须完全不存在。package成功后使用v8已通过的`build_full_ablation_stage2_feature_cache.py`参数模板补齐缺失feature identity。

每个identity使用独立、不可覆盖输出目录和日志。package按最多8行一波提交，不一次性预提交48行；每波结束读取失败日志的标准化最终异常行。P0或两个不同identity在产物发布前出现相同非空确定性异常指纹时停止后续dispatch；summary记录`launched/completed/succeeded/failed/not_launched/exception_fingerprints/systemic_stop`，不按任何性能指标停止。首批核对runner PID、worker数、GPU映射、日志增长、package seal/feature manifest计数和异常指纹。正式states启动前必须达到package、feature、sidecar、registry、seal全集闭合。

## package补齐结果

Git commit=`af0eefa6`的v9 package controller、v4 50行source summary和launch artifact已同步到release外置artifact目录，三文件远端SHA256分别为`bd74995ed0a04fbbab06519038e706a8e3bf0549501fee418eedc02265a08fa0`、`db4fbf825309c3bce123069ec0137654ff91e4efd14e5a8b1586fc6cb2ea514a`、`11659bcde18c0d3b6d89ab6d32ce7d2b9331ef0d7f61370f13167d793ba4daa3`，与本地一致。远端compile、50行共同v2 binding可读性、v5两份seal可读性及fresh input/log根检查均PASS。

runner PID=`983142`，每波8个CPU worker。第一波8/8成功，无异常指纹；最终`launched=48`、`completed=48`、`succeeded=48`、`failed=0`、`not_launched=0`、`exception_fingerprints={}`、`systemic_stop=false`，耗时18.65秒。新建48个package seal，加v5复用2个后达到50/50；runner已退出，未占GPU。

## feature补齐冻结配置

补齐99个缺失feature extraction calls；每次调用固定发布Stage2-A/B/C三个scope cache，因此新增297个scope caches、594个物理文件；再复用v8 smoke的3个scope caches后达到300个total scope caches，即A/B/C各100个。GPU slots固定为`[1,2,2,3,3,4,4,5,5,6,6,7,7]`：GPU0不使用，GPU1仅增加1进程，GPU2-7各最多2进程。每波13行，波后执行相同系统性异常停派门；不读取性能指标。driver log启动前必须不存在，并由shell `noclobber`保护，误重跑不能截断。

states的正式计数口径与物理cache数量分开：100个Stage2-B combo triples跨三个arm复用并服务300个逻辑row；每个receiver/method只选K=10的Stage2-A cache作为canonical A，再生成25个Stage2-A scoring sidecar服务25个逻辑row；共325个逻辑row。K=1/2/5伴生的75个Stage2-A cache不计为canonical Stage2-A identity。

## 完成后回填

回填package/feature/sidecar/registry/seal计数、精确runner命令/PID/GPU映射、first-wave健康状态、正式states启动证据及最终artifact状态。完整结果前不作性能结论。
