# Phase1 HCF-DG A0–A5快速源域LORO矩阵

## 预登记

- Run ID：`phase1_hcfdg_a0a5_loro2_seed3_u4000_20260830_r1`
- 当前状态：`LOCAL_VERIFIED`
- 初始集成提交：`49bd116b7af97abacb287113cb389c64b6edf42c`
- P1定点修复提交：`23e8255bca51e8fff4e3603e0df5eb4514ea6d31`
- N607布尔mask兼容修复后的正式代码提交：`7bcc5f73246c642a6cfe4983a3473f490d349ba2`
- Git分支：`codex/phase1-hcfdg-20260830`
- 方法范围：报告定义的HCF-DG V1快速筛选A0–A5；A6–A12不在本run中启动。
- 协议范围：仅Phase1 source-only域泛化；禁止访问Phase2 capsule、target receiver、support、query、truth、target prototype或目标统计。
- 数据：`Dataset_WigSig/ManySig.pkl`；source receiver集合`1,3,4,6,8`；训练日期`day1/day2/day3`；source角色`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- source LORO：fold1为中心receiver，fold8为最远receiver；每行训练时严格排除heldout receiver，训练receiver为其余4个source receiver；最终只在heldout source receiver的day1/2/3上零适配评估。
- 种子：`392001/392002/392003`。
- 预算：每行精确`4000 optimizer updates`。
- 矩阵：`6 candidates×2 folds×3 seeds=36 rows`。
- 训练视图：HCF-DG专用入口采用报告定义的70%clean+30%`mixed_orbit`单前向；不修改旧ADV3B02/ADV3B03入口。
- batch：A3–A5采用`6 TX×4 domain×4 sample=96`矩形episode；receiver/day/channel leave-out比例为`0.65/0.225/0.125`，support完整排除query因素。
- GPU：N607 GPU0–7；dispatcher对每张GPU使用2个并发槽，绝不超过2个本run训练进程/GPU。
- 选模边界：本run只产出source-only证据，低性能不停止健康训练，也不触发目标测试、调参、重训或选择性重跑。

## 候选

|候选|冻结定义|
|---|---|
|A0|ADV3B02闭集精简双分支控制；关闭FastTrust/open/unknown和旧辅助loss|
|A1|`single_parameter_matched`参数量控制|
|A2|单identity主干+48D receiver/day/channel环境编码器|
|A3|A2+矩形batch|
|A4|A3+普通LODO原型分类|
|A5|A4+rank-4公共—特定低秩头|

## 本地验证

- 聚焦与回归测试：`142 passed`。
- Python编译：launcher及HCF-DG config/sampler/satellite/model/losses/trainer/metrics全部通过。
- 真实checkpoint smoke：A0和A4均完成1 update；P1修复后A4最新smoke位于`E:\type10-7\local_artifacts\phase1_hcfdg_smoke\A4-F8-S392002-smoke3`，checkpoint为4,993,633字节，严格重建并分别产生clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`JSON和log，终态`ARTIFACTS_COMPLETE`。
- 已保留两个早期技术失败partial artifact：A2 smoke1为小型数据下`drop_last`空loader；A2 smoke2为CPU generator与CUDA增强器不匹配。两处均已定点修复，未删除失败证据。
- 独立P0/P1审查结论：`P0=0、P1=2`。P1-1为合法路径名包含`phase2`时被整串token扫描误拒；P1-2为逐样本Bernoulli及A3–A5整域mask不能固定每batch 70/30。提交`23e8255b`已定点修复：只检查显式CLI角色option；每个96样本batch固定29个satellite位置；channel episode按实际增强标签重建domain/query/support。原reviewer定点复审结论为`RESOLVED`，未重新全量审查。

## 发布前只读证据

- 2026-08-30 14:46–14:47 CST使用普通N607账户直连preflight通过；项目根、固定Python和`ManySig.pkl`可见。
- GPU0–7均为1MiB、0%利用率，`nvidia-smi`无compute app；目标训练进程计数为0。
- 目标release根、正式run根和dispatcher日志路径均为`ABSENT`，满足不可覆盖要求。
- 旧归档均保留但不发布。正式release归档大小5,461,945字节，本地/远端SHA256均为`3F379910DB1FAA5486C18E59B73A6A9DBC1A81258A96711857432FC411EDC3E6`；归档内已读回launcher、HCF-DG trainer和真实`SSDG/train_ssdg.py`，远端编译通过。解包时仅出现服务器时钟较归档时间慢约13秒的mtime warning，不影响文件或编译。
- N607 smoke r1在Python前因`noclobber`预创建日志后再次重定向而退出1；run根未创建，0字节日志保留。r2真实进入Python后暴露N607旧PyTorch无法推断`numpy.bool` dtype，退出1且无checkpoint/GPU残留，partial根保留。提交`7bcc5f73`改为显式Python`bool`列表和`torch.bool`构造。
- N607 smoke r3使用A4、fold1、seed392002、1 update、GPU0退出0；`final_hcfdg.pt`为4,993,292字节，clean与三种LEO场景均严格重建，missing/unexpected/shape mismatch均为空，终态`ARTIFACTS_COMPLETE`；完整训练日志为空且GPU已释放。1-update准确率仅为启动闭合证据，不作性能判断。
- 首次扩展资源探针被本地PowerShell提前展开远端语法并仅返回远端解析错误；未发生远端写入。随后使用无本地变量展开的短命令重新核对并取得上述有效证据。

## N607路径与命令

- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release归档：`E:/type10-7/local_artifacts/phase1_hcfdg_release/phase1_hcfdg_7bcc5f73.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/incoming/phase1_hcfdg_7bcc5f73.tar.gz`
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_7bcc5f73`
- 正式run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hcfdg_a0a5_loro2_seed3_u4000_20260830_r1`
- 正式日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hcfdg_a0a5_loro2_seed3_u4000_20260830_r1.dispatcher.log`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`

N607 smoke固定为A4、fold1、seed392002、1 update、GPU0，并要求strict final checkpoint及四场景artifact全部闭合。PASS后正式命令为：

```text
nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_7bcc5f73/code/scripts/launch_phase1_hcfdg_matrix_20260830.py --formal --run-id phase1_hcfdg_a0a5_loro2_seed3_u4000_20260830_r1 --stage quick --folds 1,8 --gpus 0,1,2,3,4,5,6,7 --code-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_7bcc5f73 --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hcfdg_a0a5_loro2_seed3_u4000_20260830_r1 --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl
```

## 直接技术停止规则

只有以下情况允许停止对应run并保留partial artifact：Phase1数据越权；错误candidate/fold/receiver/day/seed/update；输出路径冲突；错误checkout/release；主命令不能启动；无法产生prediction/evaluation闭合；同一确定性pre-prediction异常至少重复2行；进程归属不清并可能影响无关任务。低性能、收敛慢或中间指标差均不允许停止、重启或热补丁。

## 预期artifact

每行目录`<run-root>/<candidate>-F<fold>-S<seed>`必须包含：

- `final_hcfdg.pt`，含candidate、fold、seed、source split、精确update、runtime重建参数和推理头边界；
- `metrics.csv`、`metrics.jsonl`；
- `eval_clean.json/log`；
- `eval_leo_clear_weak.json/log`；
- `eval_leo_low_elev_weak.json/log`；
- `eval_leo_rain_weak.json/log`；
- `ARTIFACTS_COMPLETE.json`。

run根在36行全部进入终态后写入`final_status.json`。训练完成不等于实验完成；缺少任一严格评估场景时不得标记`ARTIFACTS_COMPLETE`。
+
## 正式启动

- 启动时间：N607约2026-08-30 15:05 CST。
- 状态：`RUNNING`。
- dispatcher PID：`2172360`；PPID1；CWD=`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_7bcc5f73`；cmdline精确绑定正式release、run ID、36行quick stage、fold1/8、GPU0–7和ManySig source路径。
- `plan.json`为12,887字节；首波创建16个row目录。
- 首波16个直属worker与dispatcher绑定正确；GPU0–7各2个本run Python训练进程，没有超过每卡2行。
- 启动30秒快照：GPU利用率69%–90%，显存约1,427–1,429MiB/卡；无`TECHNICAL_FAILURE.json`，dispatcher仍存活。首个update尚未写出metrics，下一次短连接复核增长。
- 启动74秒复核：dispatcher及16个worker仍存活，GPU利用率61%–89%，无技术失败。trainer按设计在每行4000 updates结束时一次性写`metrics.csv/jsonl`，因此运行中dispatcher日志与metrics可保持空/缺失；以精确进程绑定、GPU计算、elapsed和row目录状态监控，不据此停止健康run。
- 低频heartbeat监控ID：`hcf-dg-a0-a5`，每30分钟短连接只读检查；36行闭合后完整解析全部artifact并发布source-only A0–A5结论。
