# Phase1 GeoSat Lite四臂首发报告

状态：`ANALYZED / NO_SINGLE_ARM_PROMOTED`

目标模式：`GOAL_MODE=ACTIVE`

快速通道：`PHASE1_G0_FAST_RELEASE`

版本承载：`E:\type10-7`根目录不是Git仓库；本报告的精确Git镜像位于隔离worktree同相对路径。

## 1.实验身份

| 字段 | 值 |
|---|---|
| run ID | `phase1_geosat_lite_4arm_20260808_v1` |
| 日期 | 2026-08-08 |
| 主Agent | `/root` |
| 唯一N607 runner | `/root/n607_geosat_lite_runner`（Luna/max） |
| 目标 | 用最小四臂验证角度几何与clean→LEO一致性对Phase1高泛化表征的独立及联合贡献 |
| 实现commit | `aaaf59ca4ce2e7137dea98cb1aa1254f23ce0f1c` |
| 独立复核 | `STATUS=PASS; P0=0; P1=0; ALLOW_RELEASE=true` |

## 2.假设与比较

`P1-GeoSat-Lite`不新增模型分支，只在ADV3B02结构/closed-SSDG配方上加入两个可分离因子：known-only角度几何和单向clean→LEO KL。D相对A的改善必须由B、C和交互同排解释，不能把所有差异归因于一个组件。

| arm | `lambda_open_world_feat` | `lambda_sat_cons` | GPU |
|---|---:|---:|---:|
| A_ADV3B02_Z0 | 0 | 0 | 0 |
| B_ANGULAR_Z0 | 0.0024 | 0 | 1 |
| C_LEO_CONS_Z0 | 0 | 0.10 | 2 |
| D_GEOSAT_LITE_Z0 | 0.0024 | 0.10 | 3 |

## 3.数据与方法锁

- 数据：`Dataset_WigSig/ManySig.pkl`，source-only；N607文件hash在preflight/landing后补记。
- TX角色：train=`14-10,14-7,20-15,20-19`；held-known=`6-15`；proxy-unknown=`8-20`。
- 训练入口先按TX过滤再执行L/U/V，held TX不进入训练、source physical-validation或逐epoch输出。
- 划分：`0.07/0.63/0.30`，`rho_label=0.10`；seed=`7281105`，sat view seed=`9281105`。
- 四臂均`from_scratch=true`；历史六TX checkpoint只用于本地无query结构smoke，不加载进本轮训练。
- 120 epochs；final-only checkpoint；三种`leo_*_weak`；proxy/mixup/source-episode/direct-metric/EVT/gradient-surgery关闭。
- 本轮是4/1/1开发screen，不是六类最终deployment bundle，也不是Phase3真实unknown结果。

## 4.本地变更与验证

| 文件 | 目的 |
|---|---|
| `code/SSDG/train_ssdg.py` | TX互斥训练视图、连续重编号与receipt |
| `code/tests/test_phase1_tx_partition.py` | 角色互斥、缺失TX、CLI与过滤负测 |
| `code/scripts/launch_phase1_geosat_lite_4gpu_20260808.sh` | 不可覆盖四臂启动 |
| `analysis/phase1_geosat_lite_design_20260808.md` | 唯一快速通道方法锁 |

验证：

- `ssr-gpu` focused tests：22项通过。
- 追加focused split/loss测试：15项通过。
- launcher：`bash -n`通过；独立复核确认75个CLI参数可解析。
- 4类随机初始化no-query smoke：`tx_logits=[2,4]`、`z_id=[2,160]`、finite=true、query_input_count=0。
- Git worktree在实现提交后clean。

## 5.N607交接

| 字段 | 冻结值 |
|---|---|
| Conda/Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 项目根 | `/home/szu2070436088/2510044040/CV-SincNet` |
| release根 | 由runner按commit/hash建立非覆盖目录 |
| run root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_geosat_lite_4arm_20260808_v1` |
| log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_geosat_lite_4arm_20260808_v1` |
| launcher | `code/scripts/launch_phase1_geosat_lite_4gpu_20260808.sh` |
| GPU | 0/1/2/3，各一臂 |
| 预期产物 | 每臂`final_ssdg.pth`、`latest_ssdg.pth`、`metrics_epoch.csv/jsonl`及终态receipt；总`pids.tsv/completion.tsv` |

runner exact command在release路径/hash确定后补记，形式固定为：

```text
nohup env RUN_ID=phase1_geosat_lite_4arm_20260808_v1 CODE_ROOT=<release>/code bash <release>/code/scripts/launch_phase1_geosat_lite_4gpu_20260808.sh > <log_root>.launch.out 2>&1 < /dev/null &
```

### 5.1实际落地与启动证据

状态：`ARTIFACTS_COMPLETE`。

- release commit：`5f01bdfb1446603b393e5d5c89643e6f129f61c6`。
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_geosat_lite_4arm_20260808_v1_5f01bdfb`。
- Git archive SHA256：`train_ssdg.py=2fdac77aa38a2c5bdede3fc47f23047e33df3fb0e13f8bdaa19243ec7376435f`；`launcher=42c1aea7c2aa12a7cced8f9968da58d0943a2ebead29c5037979c0bdfb900e03`；`test=8fa2f8cea16de3b552201cd95f5a7879a573a0b3529cdf60c4793eaa7c554240`。与Windows工作树hash差异仅来自Git属性/换行口径，远端字节与固定commit归档一致。
- exact command：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_geosat_lite_4arm_20260808_v1_5f01bdfb/code && nohup setsid env RUN_ID=phase1_geosat_lite_4arm_20260808_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_geosat_lite_4arm_20260808_v1_5f01bdfb/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_geosat_lite_4arm_20260808_v1_5f01bdfb/code/scripts/launch_phase1_geosat_lite_4gpu_20260808.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_geosat_lite_4arm_20260808_v1.launch.out 2>&1 < /dev/null & echo $!
```

| 进程 | PID | GPU | 首波状态 |
|---|---:|---:|---|
| launcher | 3494462 | N/A | CWD与release绑定，PGID/SID=3494462 |
| A_ADV3B02_Z0 | 3494468 | 0 | E013，约2214MiB |
| B_ANGULAR_Z0 | 3494470 | 1 | E012，约1996MiB |
| C_LEO_CONS_Z0 | 3494472 | 2 | E011，约2242MiB |
| D_GEOSAT_LITE_Z0 | 3494474 | 3 | E010，约2454MiB |

四臂均出现`CONFIG-RUN/DATA/LOSS/SAT/TELEMETRY`与`SSDG-TRAIN`，错误指纹0/4；每臂已生成`metrics_epoch.csv`与`metrics_epoch.jsonl`。启动SSH命令超时后未重启，runner先用远端证据确认同一进程已运行，再关闭本地PID25020并确认TCP22无残留。


二次健康复核时四臂仍为LIVE：A/B/C/D分别到E024/E022/E021/E018，日志持续增长，错误指纹仍为0/4；GPU4–7空闲，短连接结束后本地SSH进程与N607/桥接TCP22连接均为none。

四臂最终均完成E120/120，训练日志错误指纹0/4，`train_skipped_nonfinite_loss=0`且`train_skipped_nonfinite_grad=0`。每臂均落盘`final_ssdg.pth`和120行JSONL/121行CSV；`latest_ssdg.pth`因冻结的`checkpoint_selection=final_only`未生成，未伪造副本。四个exit=8均对应脚本显式终端回执`NON_PROMOTABLE_P0_DISABLED`，不是OOM、异常或不完整训练。launcher及run-owned PID均退出，GPU0–7释放，SSH无残留。

### 5.2冻结后source-only held-TX审计锁

该审计是设计中已预注册的“一次冻结后审计”，不是新训练或新方法。四个`final_ssdg.pth`只读，公式和阈值规则在任何held-TX结果前固定：

- exporter：现有`export_spaceborne_features.py`；`z_id+tx_logits`；clean view；source days=`0,1`，source receivers=`0..6`，每TX最多400条；seed=`7281105`；
- known/calibration TX：`14-10,14-7,20-15,20-19`，role=`source`；
- held-known TX：`6-15`，role=`target_old`，只作第二未见TX诊断；
- proxy-unknown TX：`8-20`，role=`proxy_unknown`，作为主要proxy审计；
- evaluator：现有`eval_phase1_logits_open_set_reject.py`；只从source正确分类样本冻结`confidence Q0.05`、`margin Q0.05`、`energy Q0.95`；held TX不参与阈值、公式或模型更新；
- primary输出：source known full accuracy、proxy FAR、AUROC、safe rejection；secondary输出：把held-known作为另一未见TX的同公式诊断；
- 四臂同输入、同样本上限、同量化规则。审计通过或失败均不触发重训、fallback或阈值重选。

本地复用测试：`test_phase1_logits_open_set_reject_eval.py`、`test_phase1_open_set_reject_eval.py`和真实checkpoint loader共5项通过。

审计实际完成：四个checkpoint均严格加载，missing/unexpected/skipped=0；每臂固定导出source=1600、target_old=400、proxy_unknown=400，TX角色互斥。8次评估均exit=0；28个小型JSON/CSV/日志/manifest证据已回收，逐文件SHA256与远端清单一致，未下载NPZ或checkpoint。导出进程未预写退出码receipt，故清单如实标记`INFERRED_SUCCESS_NO_EXIT_RECEIPT`；完整NPZ、正常退出进程和零错误日志共同证明导出完成，但不补造exit code。

## 6.健康与停止规则


启动后核对launcher PID、四个child PID、CWD/cmdline、GPU映射和日志增长。仅在P0协议/覆盖风险、错误checkout/hash、query/held-TX泄漏、CUDA OOM或至少两个不同arm在产生训练telemetry前出现同一确定性异常指纹时，停止该run的后续工作并保留所有partial artifacts。不得因准确率、floor、FAR或其它性能值中止。

## 7.完整同排结果

### 7.1训练终行与LEO弱信道

| arm | 机制 | TX split | seed/epoch | source val acc | LEO mean | LEO floor | source-val H | best val@epoch | final epoch time | verdict |
|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| A_ADV3B02_Z0 | closed基线 | 4/1/1 | 7281105/120 | 97.262% | 61.125% | 59.446% | 73.792 | 97.899%@E117 | 18.90s | 基线 |
| B_ANGULAR_Z0 | known-only角度几何 | 4/1/1 | 7281105/120 | 97.065% | 60.726% | 59.214% | 73.556 | 97.935%@E118 | 19.52s | LEO无增益 |
| C_LEO_CONS_Z0 | clean→LEO单向KL | 4/1/1 | 7281105/120 | 97.268% | 70.192% | 68.833% | 80.617 | 97.595%@E117 | 19.45s | LEO最强，但非开放世界晋级 |
| D_GEOSAT_LITE_Z0 | 角度几何+单向KL | 4/1/1 | 7281105/120 | 97.280% | 70.022% | 68.893% | 80.662 | 97.488%@E108 | 20.82s | 与C近似，未形成互补 |

相对A，C的LEO mean/floor分别提升`+9.067pp/+9.387pp`；D为`+8.897pp/+9.446pp`。B单独为`-0.399pp/-0.232pp`。D相对C仅`-0.171pp/+0.060pp`，LEO交互很小。

### 7.2冻结source-only开集审计

所有行使用同一source-only阈值规则；known是用于校准的source样本，因此known数值是开发诊断，不是跨域正式成绩。proxy和held TX从未用于阈值、训练或选模。

| arm | source closed acc | source coverage | source full acc after reject | proxy FAR↓ | proxy AUROC↑ | held-known FAR↓ | held-known AUROC↑ | 5% FAR门 | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A_ADV3B02_Z0 | 97.125% | 90.313% | 89.688% | 53.500% | 0.5505 | 46.000% | 0.6185 | FAIL | 基线拒识弱 |
| B_ANGULAR_Z0 | 96.688% | 89.063% | 88.813% | 38.250% | 0.5774 | 21.250% | 0.6071 | FAIL | 相对proxy FAR明确改善，但远未达标 |
| C_LEO_CONS_Z0 | 96.625% | 88.813% | 88.125% | 66.000% | 0.5176 | 44.500% | 0.6050 | FAIL | LEO增强损害proxy拒识 |
| D_GEOSAT_LITE_Z0 | 97.188% | 90.188% | 89.750% | 79.250% | 0.5106 | 60.750% | 0.6087 | FAIL | 联合训练产生强负交互 |

B相对A把proxy FAR降低`15.25pp`，同时source full accuracy仅下降`0.875pp`，满足“相对基线产生明确正信号”，但`38.25%`仍不具备真实部署拒识能力。C相对A把proxy FAR提高`12.50pp`；D相对A提高`25.75pp`。对proxy FAR，联合交互`D-B-C+A=+28.50pp`，表明两个目标在同一读出路径上发生冲突，而不是互补。

### 7.3证据与异常边界

- 四臂checkpoint SHA256、训练metrics、完整日志和审计逐样本score表均在`artifacts/`清单中；本地未复制checkpoint。
- 所有审计FPR95均为1.0，且没有单臂通过`unknown_FAR<=5%`，不能把拒识率或AUROC包装成Phase3真实unknown结果。
- 日志中的字符串`nan`来自未启用遥测字段占位；训练计数明确为零非有限loss/gradient，不能据此误判数值崩溃。
- 本轮仅4个source-known TX、1个held-known TX和1个proxy TX、单seed；结论只支持机制筛选，不支持最终deployment bundle。

## 8.风险与后续

本轮结论为`NO_SINGLE_ARM_PROMOTED`：C证明单向clean→LEO一致性是强泛化组件；B证明known-only角度几何包含独立proxy拒识信号；D证明把两者压到同一表征/读出路径会破坏开放集边界。下一轮不增加receiver/day/channel对齐、EVT、动态门控或复杂审计，优先验证“分类读出与拒识读出解耦”的最小双读出方案：LEO分类路径继承C，开放集路径继承B，分别输出类别证据和拒识证据，再以固定source-only规则组合。该方案先做小型同输入实验，不直接扩成六类200epoch正式bundle。完整Phase3 v2/CARE-PoE设计继续延期，不阻塞Phase1实验迭代。
