# v7 N607 Runner Handoff

## 技术终态

`ARTIFACTS_COMPLETE / READY_FOR_PRIMARY_ANALYZER`

该状态只表示冻结的truth-free技术产物、truth sidecar路径和receipt已闭合；runner没有读取或解释任何性能字段。

## 冻结身份与落地

- run ID：`d92_e0_continuous_session_k10new5_20260817_v7`
- science commit：`75fcf33b9c1d77fbb1ba83bf555209944ccd6d32`
- release commit：`e9168973b15f45fea02756fd01407517a4d3a4c4`
- archive：`d92_e0_continuous_session_v7.tar.gz`，171137B，SHA256=`791b1eb3f243b93371cec22f3bb5aabace8e6de57fb4a94f2890d1b9087e1054`，31成员，安全tar验证通过。
- launch：`launch.sh`，2261B，SHA256=`6845226173379b07cf763d0dd9e1ff888a34d24d0361304189fc7dd2e7597775`，`bash -n`通过。
- 远端Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CUDA可见。初始run/log根、PID与本地retrieval均为ABSENT，GPU0–4无compute占用。
- 唯一提交命令：`cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v7 && nohup bash ./launch.sh >./launch.out 2>./launch.err </dev/null &`。

## 启动连接与技术健康

- 启动SSH客户端未正常退出；唯一残留PID 32276与本次启动命令及N607 TCP22连接绑定。正常结束失败后才对该一个本地客户端执行强制结束。随后只读远端核验确认run已落地；没有第二次启动。
- 首次完成核验时run-owned PID=0、GPU compute=0，日志技术异常标记为`NONE`。
- 5个formal `job_receipt.json`均为truth-free预测完成状态；共6份`prediction_manifest.json`（smoke+5 formal）。
- query truth/fit/update/selection/role/quota/global-reassignment访问字段均为false。
- smoke只比较`batch_5`与`singleton_forward`；formal每outer比较`batch_5`、`singleton_forward`、`singleton_reverse`、`chunk_2_2_1`。已核验终端closure状态均为`STRICT_EQUAL`。

## 保全与取回

| artifact | count |
|---|---:|
| `matrix_manifest.json` | 1 |
| `prepared_manifest.json` | 5 |
| `delta_receipt.json` | 5 |
| `prediction_manifest.json` | 6 |
| `job_receipt.json` | 5 |
| `execution_receipt.json` | 246 |
| `fit_audit.json` | 246 |
| `resource_audit.json` | 246 |
| `COMMIT.json` | 251 |

- 完整取回根：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v7`。
- `run_root`含release archive、runtime/source、output、launch、`launch.out`与`launch.err`，共1420文件；`log_root`共10文件。
- 已复制5份`truth_sidecar.json`至`truth_sidecars/jobs/<outer>/offline/scorer/`。truth内容没有打开或解析，`content_read=false`。
- 每个outer的`job_receipt.json`已随`run_root`取回，作为receipt路径闭合；预注册scorer根未发现另一个独立命名的receipt文件，因此没有猜测或额外复制文件。

## 分析边界与清理

- 未读取accuracy、H、BA、floor或其他性能字段，未运行analyzer，未调参或重试。
- 资源收据由主代理在truth-last阶段按冻结规则处理；runner未以wall/peak作健康停止。
- 最终本地`ssh.exe`=0、`scp.exe`=0、N607 TCP22 `ESTABLISHED`=0；远端run-owned PID=0、GPU compute=0。
