# 指定ADV3B02 checkpoint的A1续训

用户更新：使用已独立评分的ADV3B02 E200最终checkpoint继续训练，seed为392005；取代此前从零重训CORE90的依赖链。该基线的target结果仅作为用户指定起点的背景，不用于本轮loss、超参数或checkpoint选择。此次为用户指定起点后的执行优化配对实验，不是对未见确认集的独立方法选择证据。

## 数据核对

直接读取指定checkpoint的args及split_info，与本轮解析后配置比较：

|项目|checkpoint与续训共同配置|
|---|---|
|数据|`Dataset_WigSig/ManySig.pkl`，equalized=1|
|seed|392005|
|source receiver|1,3,4,6,8|
|source day|1,2,3|
|target receiver|0,2,5,7,9,10,11|
|target day|0,1,2,3|
|split|tx_rx_day_1_7_2；legacy_l_u_v|
|L/U/V|0.07/0.63/0.30；6300/56700/27000|
|TX与输入|0-5共6类；256点|
|采样|random；各combo cap=0；guard_gap=8|
|增强seed|sat_seed=2027、sat_view_seed=2027；与原checkpoint一致|

数据配置一致。审计最初发现模型差异：原A1为M/lite_c/dual，所选权重为M/lite_d/dual；两组续训统一改为lite_d。权重检查逐tensor准确加载原有参数，唯一新增随机参数为A1的daot_nuisance_head四个tensor。该头的零权重梯度行为继续保留，不将其移除或声称已激活有效监督。

## 冻结矩阵和执行

run_id=`a1_fast_selected_adv3b02_s392005_20260908_r1`。

两组使用同一`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`作为baseline和teacher。核对epoch=200、大小15039647字节、mtime=2026-09-03 08:00:52 +0800。

|行|GPU|追加预算|执行差异|
|---|---:|---:|---|
|A1_REFERENCE|4|E200|原教师路径|
|A1_FAST_SEQUENTIAL|5|E200|逐视图身份教师、尺度/日志批量读回、mean冗余中间量跳过|

权重初始化续训，优化器、随机状态及A1阶段日程重新开始，不声称从旧E200 optimizer状态精确resume。两组共享seed392005及原A1其他训练参数。旧训练入口不接受never，故使用`test_eval_policy=interval_final`、start=999999及interval=0，且`muse_external_final_eval=true`、final-only；source兼容指标使用clean_val_tx，不读取target作训练选择。batch教师仍暂缓，原EMA缓存缺陷仍保留，不混入Stable修复。

启动入口：`python code/scripts/run_a1_fast_selected_adv3b02.py --project-root /home/szu2070436088/2510044040/CV-SincNet --run-id a1_fast_selected_adv3b02_s392005_20260908_r1`。工作目录为对应不可变release；Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python。

原r3只终止已核对PID/CWD/cmdline/父子关系的旧dispatcher和CORE90，状态记SUPERSEDED_BY_USER，保留全部产物。终止旧dispatcher后不会再排队启动旧checkpoint分支。不干预其他任务。

输出：`runs/<run-id>/<row>/`；日志：`logs/<run-id>/`；拒绝已存在根目录，每GPU最多两训练进程。第一步真实指定checkpoint无query GPU检查通过后启动两行。路径/权重绑定错误、异常/OOM/非有限技术故障停止所属行，不按性能停止，不自动重试。

E200后复用已经验证的`target_inputs`无标签包。复用提交`0cc19956`中已修复NumPy兼容问题的predictor、scorer和truth_last实现，但运行在本轮支持A1新头的模型release中；显式选择A1的tx_logits。旧评估release不支持新增头，直接调用会严格加载失败，已在发布前审查中发现并修正。新增真实lite_d+A1头完整重建及相同logits测试。prediction完成后独立scorer连接`target_truth/truth_sidecar.json`。每场景168000条，四场景672000条；输出为各行新`target_prediction`。不覆盖旧prediction，不把分数反馈训练。

预期artifact：两行E200 checkpoint、完整CSV/JSONL、GPU执行检查、四场景prediction/score及日志；最终仅标AWAITING_ARTIFACT_ANALYSIS，待全量激活、资源与性能解释。当前本地真实checkpoint CPU执行检查PASS；正式启动状态在发布读回后追加。
