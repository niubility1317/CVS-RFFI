+# Phase1 GeoSat Lite四臂首发报告

状态：`LOCAL_VERIFIED / READY_FOR_N607_HANDOFF`

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
