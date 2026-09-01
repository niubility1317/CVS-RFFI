# RIEI/DRIFT新Phase1数据包fold1/fold8首轮报告

- run_id：`phase1_riei_drift_newsplit_fold18_s392002_20260901_r1`
- 当前状态：`LOCAL_VERIFIED`
- 冻结代码提交：`77f90a3324f5ab1dea373be96576082e43cfca75`
- 本地环境/CWD：`ssr-gpu`；`E:\type10-7\github_publish\CVS-RFFI-repo`
- N607环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；不可变release目录`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_riei_drift_newsplit_fold18_s392002_20260901_r1_77f90a33`
- 数据输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_riei_drift_newsplit_fold18_s392002_20260901_r1`
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_riei_drift_newsplit_fold18_s392002_20260901_r1`
- GPU：RIEI-fold1→0，RIEI-fold8→1，DRIFT-fold1→2，DRIFT-fold8→3。
- 停止规则：固定200epoch，不按updates或性能提前终止；仅在数据/query越权、错误split/fold/seed/day/receiver、输出碰撞、错误checkout、无prediction闭合、launcher级故障或至少两行出现相同确定性异常时，精确绑定并停止本run进程树，保留全部产物。
- 预期artifact：每行`best_by_val.pt`、`metrics.json`、clean与`leo_clear_weak`/`leo_low_elev_weak`/`leo_rain_weak`逐场景结果及日志；run根包含`matrix_manifest.json`和`launch_receipt.json`。

## 冻结矩阵

| 方法 | fold | seed | L_s/U_s/V | source留出 | GPU |
|---|---:|---:|---|---:|---:|
| RIEI | 1 | 392002 | 0.07/0.63/0.30 | receiver1 | 0 |
| RIEI | 8 | 392002 | 0.07/0.63/0.30 | receiver8 | 1 |
| DRIFT | 1 | 392002 | 0.07/0.63/0.30 | receiver1 | 2 |
| DRIFT | 8 | 392002 | 0.07/0.63/0.30 | receiver8 | 3 |

统一source receivers=`[1,3,4,6,8]`，训练day1–day3。单一source验证集`V=0.30`用于checkpoint选择，不拆分其他验证角色。训练采用真实`clean+satellite`批次拼接；clean半批计算方法原损失，satellite半批只计算TX交叉熵，`lambda_sat_cls=0.68`、`lambda_sat_cons=0`。卫星辅助CE从E80开始，课程为E1–40 clear/p=0.30、E41–90 low-elev+rain/p=0.60、E91–200三场景/p=0.80。训练期不运行目标测试；冻结V选择checkpoint后，只在目标receiver全集上评估day1–day4的clean及三种LEO_WEAK，结果不得反馈选种、调参或重训。

## 启动命令

```text
PYTHONPATH=<release>/code:<release> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m cvsrffi.phase1_baseline_fold_matrix --run-id phase1_riei_drift_newsplit_fold18_s392002_20260901_r1 --project-root <release> --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_riei_drift_newsplit_fold18_s392002_20260901_r1 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_riei_drift_newsplit_fold18_s392002_20260901_r1 --python-bin /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --gpu-ids 0,1,2,3 --execute
```

## 本地验证与审查

- 变更：baseline source split默认值与独立source留出；RIEI/DRIFT真实拼接CE-only；V选优checkpoint最终重载；目标端clean/LEO静默最终评估；4行不可覆盖launcher；协议镜像和聚焦测试。
- 聚焦+相邻回归：30项通过。
- Python编译与`git diff --check`：通过。
- 独立P0/P1审查及一次原问题定点复审：`READY`。
- Git push与远端OID回读：`VERIFIED`，远端`work/cvs-active`=`77f90a3324f5ab1dea373be96576082e43cfca75`。
- release映射：本地`E:\type10-7\release_archives\phase1_riei_drift_newsplit_fold18_s392002_20260901_r1_77f90a33.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_riei_drift_newsplit_fold18_s392002_20260901_r1_77f90a33.tar.gz`；仅比较该归档一次本地/远端SHA256。

