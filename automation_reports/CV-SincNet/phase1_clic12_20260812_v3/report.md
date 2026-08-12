# Phase1 CLIC 12臂训练v3预注册与运行报告

## 1. 状态

- 实验ID：`phase1_clic12_20260812_v3`
- 当前状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED`
- 本轮是发布工程第二次且最后一次修复后的独立one-shot入口；不得恢复v1/v2，也不得在v3失败后继续堆叠launcher修补。
- v1：argparse布尔参数缺陷，12/12训练前退出，`NO_PERFORMANCE_RESULT`。
- v2：修复布尔解析后，12/12在数据构建前因缺少proxy-unknown TX集合退出，12份failure receipt、0 checkpoint，`NO_PERFORMANCE_RESULT`。

## 2. 根因与最小修复

- Phase1 TX互斥合同要求每折`source_known_train_tx`、`source_known_validation_tx`、`source_proxy_unknown_tx`均非空且两两互斥。
- v2只传了4个训练TX和1个known-validation TX，没有把六TX全集中唯一剩余TX显式绑定为source-proxy unknown。
- v3只新增冻结数组`FOLD_PROXY_TX=(14-10,14-7,20-15,20-19,6-15,8-20)`并逐折传入`--phase1_source_proxy_unknown_tx_ids`；方法、数据、loss、seed、epoch、GPU矩阵不变。
- 修复commit：`f053ea7e`（`fix: bind CLIC proxy TX partition`）。
- 本地回归现在对12条真实子命令逐条验证：argparse完整解析；每折恰好4+1+1；三角色互斥；并集恰为冻结六TX全集。
- `ssr-gpu`完整CLIC：`164 passed`；`py_compile`、`bash -n`、dry-run12均通过。

## 3. 冻结科学矩阵

F1—F6×C/G共12臂；C=`raw_phase_control`，G=`complex_local_invariant_curvature`；seed=`7281164`；40epoch；batch=128；AdamW；lr=`2e-4`；`clean CE+0.10×KL(clean-stopgrad→single-LEO)`；三source LEO weak场景；final-only；旧机制关闭；target/query/target truth/role零训练访问。

每折TX角色：

| fold | train4 | known-validation1 | proxy-unknown1 |
|---|---|---|---|
| F1 | `20-15,20-19,6-15,8-20` | `14-7` | `14-10` |
| F2 | `14-10,20-19,6-15,8-20` | `20-15` | `14-7` |
| F3 | `14-10,14-7,6-15,8-20` | `20-19` | `20-15` |
| F4 | `14-10,14-7,20-15,8-20` | `6-15` | `20-19` |
| F5 | `14-10,14-7,20-15,20-19` | `8-20` | `6-15` |
| F6 | `14-7,20-15,20-19,6-15` | `14-10` | `8-20` |

GPU映射不变：0=`F1C,F5G`；1=`F1G,F5C`；2=`F2C,F6G`；3=`F2G,F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`。

## 4. N607发布合同

- 发布commit：`f053ea7e`；必须从该commit干净Git archive发布，不得带Task7工作树。
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic12_20260812_v3_f053ea7e`
- run/log/outer：分别使用`runs/phase1_clic12_20260812_v3`、`logs/phase1_clic12_20260812_v3`和项目根`phase1_clic12_20260812_v3_outer.out`。
- v3唯一launch=1，retry=`NO`；启动前确认新路径不存在、GPU资源允许、12 warm-start存在。
- 唯一launcher文件名保持`launch_phase1_clic12_20260811.sh`，但其默认RUN_ID已冻结为v3。

## 5. 健康与后续

启动后核12 PID/CWD/cmdline/GPU、`pids.tsv`、日志增长、数据分区receipt和首个训练batch。只按既有P0/安全/系统性技术异常规则停止，绝不按性能停止。每臂预期`final_ssdg.pth`、`phase1_clic_terminal_receipt.json`和完整log。

训练完成后执行postfreeze及叠加LEO weak的目标域known与unknown盲态测试；每个指标必须包含LEO weak目标域结果，核心仍为未知类拒识与域泛化。

## 6. 运行回填

- archive SHA/bytes：待runner
- SCP/launch次数：待runner
- release静态门：待runner
- PID/GPU/日志首波：待runner
- 最终状态：待runner
