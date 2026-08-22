# Phase1 ADV3B02 SID-FFT96残差上界修正验证

## 当前状态

- run ID：`phase1_advb02_sidfft96_guarded_capfix_20260823_v1`
- 状态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- 目标：修复公共模型构建器漏传`sid_max_residual_ratio`的问题，仅重跑修正后的`S3G_SIDFFT96_GUARDED`，复用上一轮已完成的S0，不扩大矩阵。
- 科学边界：本轮仍是Phase1 source-only代理实验，不引入target/query，不声明Phase2适应、新类注册、Phase3未知拒识或真实在轨性能。

## 最小预登记

|字段|冻结值|
|---|---|
|候选/矩阵|仅`S3G_SIDFFT96_GUARDED`；S0复用`phase1_advb02_sidfft96_guarded_20260822_v1/S0_FROZEN_CORE90`|
|基座|`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|数据协议|`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，`R_s∩R_t=∅`，query/target输入计数必须为0|
|seed|`392002`|
|训练|200个epoch；仅`sid_fft96.*`可训练；`sid_max_residual_ratio=0.10`|
|必评场景|clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|本地工作区|`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\advb02-sid-capfix-20260823`|
|Git分支|`codex/advb02-sid-capfix-20260823`|
|Git提交|训练构建修复`52fc45852828f2d336fe65849394648bf2ed9265`；评估重建修复`26477fc49d513bc01f19304f629655180187aee5`|
|N607环境/CWD|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_capfix_26477fc4`|
|运行目录|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_sidfft96_guarded_capfix_20260823_v1`|
|日志目录|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_sidfft96_guarded_capfix_20260823_v1`|
|GPU|GPU7；启动前preflight为0个计算进程，GPU0–6各1个既有训练进程|
|启动命令|以release为`ROOT`，显式绑定主目录的`RUNS_ROOT`、`LOG_ROOT`、`WISIG_PKL`、`BASELINE_CKPT`和`SID_MASK_PATH`，设置`RUN_ID=phase1_advb02_sidfft96_guarded_capfix_20260823_v1`、`ONLY=S3G`、`GPU_S3G=7`，执行`bash code/scripts/launch_phase1_advb02_sidfft96_guarded_20260822.sh --only=S3G`|
|预期artifact|`final_ssdg.pth`、选择checkpoint、200行`metrics_epoch.csv/jsonl`、训练stdout、`independent_final_eval/final_eval.json/txt`、独立评估stdout|

## 停止规则

只允许因协议/query泄漏、错误stage/seed/split/scenario、输出碰撞、错误checkout、确定性重复异常、无prediction闭合、scorer连接错误、进程归属不清或预登记的系统技术故障停止。低性能不停止训练；发现性能不足只在完整评估后判定不晋级。任何停止都只能作用于该run ID绑定的精确进程树，并保留已有artifact。

## 晋级门槛

与复用S0同row比较：clean下降不超过0.3pp；三种LEO overall均值至少提升1.0pp；三种LEO Strict UDU均值至少提升1.0pp；LEO overall floor至少提升0.5pp；运行时残差上界必须为0.10且逐样本残差比不超过0.10；非SID参数零漂移。未满足则`SCIENTIFIC_FAILURE_NO_PROMOTION`，不得进入多seed或完整确认矩阵。

## 设计—实现—验证追踪

|ID|要求|实现位置|验证|状态|
|---|---|---|---|---|
|R1|公共构建器显式透传`sid_max_residual_ratio`|`code/post_stage_common.py::build_baseline_model`|训练checkpoint与评估模型运行时属性均为0.10|`VERIFIED`|
|R2|逐样本残差范数不超过原始身份嵌入的10%|`SIDFFT96Residual.forward`既有裁剪逻辑|最终checkpoint复扫全部5,880个`L_s`样本|`VERIFIED`|
|R3|真实checkpoint无query smoke且仅SID参数可训练|既有smoke与训练配置|N607 release真实checkpoint smoke|`VERIFIED`|
|R4|只重跑修正S3G并闭合clean及三种LEO_WEAK|既有guarded launcher的`--only=S3G`|N607完整artifact与同row评分|`VERIFIED`|
|R5|独立最终评估重建保持相同的0.10上界|`code/evaluation/collaborative_inference_eval.py::build_model_from_checkpoint_args`|N607最终checkpoint实际重建、严格加载与属性回读|`VERIFIED`|

## 工作区隔离说明

原检出已切换到`codex/advb02-hsid-20260823`并存在3个未提交的HSID相关文件改动。本轮从已发布提交`8c5cae87a91e567ef06c680e5f9a372ebee8e166`创建独立capfix worktree，未切换、暂存、提交或清理原分支的任何改动。

## 本地验证记录

- 基线：修复前SID相关20项测试全部通过，证明既有测试没有覆盖公共构建器透传。
- RED：新增`test_build_baseline_model_applies_sid_residual_cap`后得到运行时`0.0 != 0.1`，准确复现报告中的实现偏差。
- GREEN：补充单一参数透传后，新测试通过；SID、频谱审计、launcher及公共构建器相关21项测试全部通过。
- 新测试将SID projector权重放大后执行实际前向，4/4个样本的残差比均不超过`0.100001`。
- `py_compile`、launcher语法检查与`git diff --check`均通过。

据此，R1、R2和R5状态更新为`VERIFIED_LOCAL`；R3等待真实ADV3B02 checkpoint无query smoke，R4等待N607真实性能artifact。

## 独立P0/P1审查与定点修复

- 首次独立审查未发现P0，但发现1个P1：训练公共构建器已得到0.10，launcher调用的独立最终评估构建器仍漏传该参数，导致clean及三种LEO_WEAK可能以0.0上界评估。
- 按TDD定点复现：checkpoint参数为0.10时，修复前评估模型运行时属性为0.0。
- 最小修复：仅在评估构建器增加同一参数透传；对应测试转绿，训练/评估两条构建路径及相关SID、launcher测试共29项通过。
- 唯一一次定点复审结论为`RESOLVED`：评估构建器从checkpoint参数恢复0.10，定点测试通过，未重新扩展审查范围。
- 首版release`phase1_advb02_sidfft96_guarded_capfix_52fc4585`在审查返回前只完成归档传输、SHA核对、远端编译和dry-run，从未启动训练，现标记为`SUPERSEDED_NOT_LAUNCHED`。真实实验只能使用包含评估修复的新release。

## 发布与启动前证据

- 可启动release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_capfix_26477fc4`。
- release归档本地与远端SHA-256均为`b33b0e0d6268432a11fe72f2c7b9816a1c2053c1961f6a8d6e56c84e68becc4a`，状态`VERIFIED`。
- 远端`py_compile`和只选择S3G的完整命令dry-run均为`VERIFIED`；dry-run确认训练和独立评估均从修复release加载代码。
- 真实ADV3B02 checkpoint smoke为`VERIFIED`：`batch_role=L_s`、`query_input_count=0`、`target_input_count=0`、raw/SID初始差异为0、非SID可训练参数为0、SID可训练参数为41,280。
- 新run root在preflight时不存在；log root仅包含本轮smoke，不存在S3G训练输出，launcher仍可执行其不可覆盖检查。

## 启动健康检查

- N607服务器时间2026-08-23 02:42（Asia/Hong_Kong）启动；launcher PID为`1252254`，主训练PID为`1252274`。
- launcher CWD和训练代码均绑定`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_capfix_26477fc4`，cmdline绑定唯一run root、seed 392002、`--only=S3G`和`--sid_max_residual_ratio 0.10`。
- GPU7在启动前计算进程数为0，启动后`nvidia-smi pmon`确认PID`1252274`为GPU7计算进程；没有触碰GPU0–6的既有任务。
- 首个5秒采样恰处于日志静默窗口，训练日志保持12,474字节；随后的只读复核增长至18,167字节并闭合E2/200，因此日志增长状态为`VERIFIED`，不是启动失败。
- E2时source validation TX为98.83%，source satellite mean/floor为87.93%/86.20%；heldout test保持按预登记不在训练期执行。未发现Traceback、OOM、Killed或协议错误。

## 最终闭合状态

- 训练完整结束于E200；`metrics_epoch.csv`与`metrics_epoch.jsonl`均为200条连续epoch记录，JSONL解析错误为0。完整训练stdout共9,019行，`[EPOCH-BEGIN]`与`[EPOCH-END]`各200次；Traceback、RuntimeError、OOM、warning和执行级error均为0。
- `final_ssdg.pth`、`best_source_validation_ssdg.pth`、训练终态记录、资源记录及独立评估JSON/TXT/stdout均已生成。独立评估覆盖clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，模型状态严格加载为0个missing key、0个unexpected key。
- 最终评估checkpoint SHA-256为`305ef30a99003a291d41a4c9da7d8dbcf2bc4ac8aba7aecfb075234887485007`。文件名为`final_ssdg.pth`，其角色是`source_validation_selected_export`，内部选择epoch为E44；这与预登记的`checkpoint_selection=source_validation_only`一致，不应解释为E200末步权重。
- S0与S3G的`split_info`完全一致；除checkpoint及输出路径外，两者独立评估参数完全一致，满足同row比较条件。

## S0同row最终结果

单位均为百分比；`Δ`为S3G−S0，单位pp。

|场景/指标|S0|S3G capfix|Δ|
|---|---:|---:|---:|
|clean overall|90.1402|90.1270|-0.0132|
|clean Strict UDU|86.0900|86.1967|+0.1067|
|clean seen-day/unseen-RX|89.0533|88.9300|-0.1233|
|clean unseen-day/seen-RX|93.8095|93.7893|-0.0202|
|`leo_clear_weak` overall|78.4691|78.5078|+0.0387|
|`leo_clear_weak` Strict UDU|72.5533|72.5933|+0.0400|
|`leo_low_elev_weak` overall|75.6461|75.6877|+0.0417|
|`leo_low_elev_weak` Strict UDU|69.8633|69.8933|+0.0300|
|`leo_rain_weak` overall|75.2912|75.3201|+0.0289|
|`leo_rain_weak` Strict UDU|69.2717|69.2567|-0.0150|
|三场景overall均值|76.4688|76.5052|+0.0364|
|三场景Strict UDU均值|70.5628|70.5811|+0.0183|
|三场景overall floor|75.2912|75.3201|+0.0289|
|三场景Strict UDU floor|69.2717|69.2567|-0.0150|

## 预登记门槛判定

|门槛|实测|判定|
|---|---:|---|
|clean下降不超过0.3pp|-0.0132pp|`PASS`|
|三场景LEO overall均值至少+1.0pp|+0.0364pp|`FAIL`|
|三场景LEO Strict UDU均值至少+1.0pp|+0.0183pp|`FAIL`|
|LEO overall floor至少+0.5pp|+0.0289pp|`FAIL`|
|训练及评估运行时上界为0.10|训练记录全200行均为0.10；评估重建模型为0.10|`PASS`|
|逐样本残差比不超过0.10|5,880个`L_s`样本最大值0.100000009；超过0.100001为0|`PASS`|
|非SID参数零漂移|195/195个公共非SID状态键相等，最大绝对漂移0|`PASS`|

最终结论为`SCIENTIFIC_FAILURE_NO_PROMOTION`。修复后的0.10上界确实生效并恢复了实验可解释性，但S3G对三种LEO弱场景的平均、floor和Strict UDU增益只有0.018–0.036pp量级，远低于1.0pp/0.5pp门槛；`leo_rain_weak`的Strict UDU还下降0.015pp。因此本候选不得进入多seed或完整确认矩阵。

## SID上界、参数漂移与数值稳定性

- 训练侧：200行结构化记录的`sid_max_residual_ratio`均为0.1，`sid_adapter_only=true`、`sid_guarded_training=true`。训练期平均有效残差比由E1的0.00084升至E200的0.02949，全程epoch均值最大为0.03340（E199）。
- 评估侧：从最终checkpoint参数实际重建评估模型后，模型和`SIDFFT96Residual`模块的运行时上界均为0.1，checkpoint严格加载为0缺失/0意外。
- 逐样本复扫：对全部5,880个`L_s`样本执行最终checkpoint前向，残差比分位数Q0/Q50/Q95/Q99/max分别为0.00160/0.00820/0.01895/0.03785/0.100000009；非有限值0，超过容差0.100001的样本0。
- 参数漂移：最终checkpoint含5个`sid_fft96.*`状态键，共41,536个状态元素，其中可训练SID参数41,280；其余195个状态键与ADV3B02基座逐张量比较，缺失0、额外0、形状不匹配0、变化键0，最大绝对漂移为0。
- 非有限梯度：未出现非有限loss，但E1、E51、E85和E132各有1/45个batch触发保护性梯度跳过，共4/9,000个batch；实际优化步8,996次。训练继续闭合且最终指标有限，但这一现象说明SID projector的梯度稳定性仍未彻底消除，应作为后续候选的数值稳定性风险记录，不能写成“非有限梯度为0”。
- stdout中的`nan`仅出现在未启用或未定义的诊断槽位，例如训练期不执行heldout test、inactive direct-metric和未提供receiver-floor聚合；它们未进入训练loss或最终TX准确率。结构化训练loss、梯度汇总、SID残差比及最终准确率均为有限值。

## 训练曲线与资源

- source validation TX在200个epoch内保持98.8254%–98.8571%，没有表征坍塌。source satellite mean/floor在E44达到88.7434%/87.1508%，E200为88.5608%/86.9286%；选模因此落在E44。
- 总loss从E1的0.7124升至E118的3.5676后回落至E200的2.8166。该变化与E80后卫星CE及后续pseudo/开放世界辅助项进入总目标同步；由于source validation与卫星validation保持稳定，不能把总loss绝对值上升单独解释为训练发散。
- clip前梯度汇总最大1.9847（E81），clip后汇总最大0.9770（E97），与`max_grad_norm=1.0`一致。
- 训练资源记录：总参数1,090,945，可训练参数41,280（3.78%）；wall time 5,613.56秒（93.56分钟），200个epoch循环时间合计5,484.77秒（91.41分钟）；峰值CUDA allocated 572,168,192字节（545.66MiB），reserved 780,140,544字节（744.00MiB）。该记录是共享服务器上的吞吐观察，不是隔离延迟基准。

## 追踪闭合与后续决定

- 设计追踪共5项：`verified=5`、`deferred=0`、`rejected=0`、`blocked=0`。本轮实现与预登记的残差上界修正保持严格一致，不是近似替代。
- 最高风险剩余项是4次非有限梯度保护性跳过；它没有破坏本run闭合，但下一候选若继续SID路线，应优先约束projector梯度或学习率，而不是放宽0.10残差上界。
- 科学决定：停止S3G扩展，不发布多seed/完整矩阵。若继续前端可辨识分解路线，应改用能产生可测跨域增益的机制候选，并保持ADV3B02基座、S0同row、0.10结构上界和非SID零漂移，以避免把微小随机差异误判为机制收益。
