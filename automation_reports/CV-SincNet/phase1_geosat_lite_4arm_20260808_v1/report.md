+# Phase1 GeoSat Lite四臂首发报告

状态：`RUNNING`

目标模式：`GOAL_MODE=ACTIVE`

快速通道：`PHASE1_G0_FAST_RELEASE`

版本承载：`E:\type10-7`根目录不是Git仓库；本报告的精确Git镜像位于隔离worktree同相对路径。

## 1.实验身份

| 字段 | 值 |
|---|---|
| run ID | `phase1_geosat_lite_4arm_20260808_v1` |
| 日期 | 2026-08-08 |
| 主Agent | `/root` |
| 唯一N607 runner | 待交接的Luna/max |
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

状态：`RUNNING`。

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

## 6.健康与停止规则


启动后核对launcher PID、四个child PID、CWD/cmdline、GPU映射和日志增长。仅在P0协议/覆盖风险、错误checkout/hash、query/held-TX泄漏、CUDA OOM或至少两个不同arm在产生训练telemetry前出现同一确定性异常指纹时，停止该run的后续工作并保留所有partial artifacts。不得因准确率、floor、FAR或其它性能值中止。

## 7.结果表（待完成）

| arm | TX split | seed | epochs | known cross-RX | min known class | LEO floor | proxy FAR/AUROC | checkpoint/bundle | resource | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| A_ADV3B02_Z0 | 4/1/1 | 7281105 | 120 | pending | pending | pending | pending | pending | pending | pending |
| B_ANGULAR_Z0 | 4/1/1 | 7281105 | 120 | pending | pending | pending | pending | pending | pending | pending |
| C_LEO_CONS_Z0 | 4/1/1 | 7281105 | 120 | pending | pending | pending | pending | pending | pending | pending |
| D_GEOSAT_LITE_Z0 | 4/1/1 | 7281105 | 120 | pending | pending | pending | pending | pending | pending | pending |

## 8.风险与后续

最大风险是四TX随机初始化在120 epochs内欠拟合；该风险按冻结epoch如实观察，不通过结果延长或调参。只有四臂完整同排返回后，才决定是否以全部六个旧类、200 epochs重训正式Phase1 bundle。完整Phase3 v2/CARE-PoE设计已延期，不阻塞本实验。
