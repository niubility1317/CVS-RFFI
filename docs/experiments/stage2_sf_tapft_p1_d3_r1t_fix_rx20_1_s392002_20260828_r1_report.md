# SF-TAPFT P1 D3 R1-T定点修复实验报告

## 1.最小预登记

- run ID：`stage2_sf_tapft_p1_d3_r1t_fix_rx20_1_s392002_20260828_r1`
- 当前状态：`LOCAL_VERIFIED`
- 实现commit：`24493929ffce87f371bf036b219cc733b75d05eb`
- 候选：D3 R1-T；327/0/0固定步；all time norm；4-fold support-only OOF温度；原位full-support refit；delta v2 only
- 数据：`p2_min_v1/VALIDATED_ONCE`；capsule=`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`；split=`stage2b-rx20-1-seed713101-before-support-prefix`；旧6类K=10，共60条support
- query边界：OOF温度和full-support refit只读support，不读取query、truth、role或source；本run只产生适配delta
- 触发原因：原矩阵D3虽配置OOF温度，但旧deploy入口未执行；旧D3标记`METHOD_MISMATCH_NO_R1T_RESULT`并保留，不覆盖

## 2.验证、命令与路径

- 本地`ssr-gpu`相关测试117项通过；D3定点P0/P1复审PASS
- N607环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，PyTorch2.1.0+cu121
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_p1_d3_r1t_fix_rx20_1_s392002_20260828_r1`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_p1_d3_r1t_fix_rx20_1_s392002_20260828_r1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_p1_d3_r1t_fix_rx20_1_s392002_20260828_r1`
- GPU：0
- 命令：`python code/scripts/run_sf_tapft_slim_matrix_row.py --matrix configs/stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828.json --row-id D3 --mode deploy --deployment-inplace --delta-only --output-dir <run-root>/support/D3 --device cuda:0`
- 预期artifact：`selection.json`、`sf_tapft_delta_bundle.pt`、stdout/stderr、GNU time
- 停止规则：仅协议/query泄漏、错误数据句柄、输出碰撞、错误checkout、确定性异常、无合法delta或进程归属不清；不得因低性能停止

