# ADV3B02-DAOT-STN-RX-V2 P1～P5实验报告

## 1.状态

- run ID：`phase1_adv3b02_daot_stn_rx_v2_p1_p5_manysig_s392005_20260903_r1`
- 当前状态：`RUNNING`
- Git分支：`codex/adv3b02-daot-stn-v1-20260901`
- 首次实现提交：`c34e812f3236d1e167fc4d2508718db9104341a5`
- P1/P2发布入口提交：`4e2acba7b504c8c282f073ae40cdc64feb1bd905`
- 本报告对应的扩展提交：`6d2276ca22f552acc0d6b6dd7a6e2b7a4869c3f2`；实际release提交：`e498fc69f6371c06c1cda71df41ef1a6b72efc47`。
- 发布边界：不含ADV3B02基线；不执行非LEO_WEAK场景；不启动效率E1或高风险子空间R1。
- 用户追加授权：P1/P2之外，同时启动P3/P4/P5扩展确认；并明确不受默认GPU线程数限制。

## 2.预登记矩阵

|row|GPU|seed|相对前一行的唯一新增机制|冻结权重|
|---|---:|---:|---|---|
|V2-P1|1|392005|锚定式非对称orbit teacher，两fresh+Temporal Orbit Memory|`tangent/route/RX/tail=0`|
|V2-P2|7|392005|方向注册表、稳定弦长敏感度与预算化tangent|`λ_tangent=0.035`|
|V2-P3|0|392005|随机单TX方向干预与Directional Jacobian Routing|`λ_route=0.05`|
|V2-P4|2|392005|Source-only Style Bank条件下的TX条件RX对齐|`λ_RX=0.075`|
|V2-P5|4|392005|连续U可信度参与的receiver×channel尾部CVaR及独立调度|`λ_tail=0.10`|

所有行共同关闭旧`nuisance`、旧`fingerprint keep`和选择性subspace，即`λ_nuisance=λ_fingerprint=λ_subspace=0`。

数据协议：ManySig equalized；`split_mode=tx_rx_day_1_7_2`；source RX=`[1,3,4,6,8]`、day=`[1,2,3]`；target RX=`[0,2,5,7,9,10,11]`、day=`[0,1,2,3]`；`L_s/U_s/V=6300/56700/27000`；seed=`392005`；200epoch。

最终评估白名单：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。

## 3.本地验证

- 聚焦测试：`tests/test_adv3b02_daot_stn.py`、`tests/test_daot_rx_v2_control.py`、`tests/test_adv3b02_daot_stn_rx_v2_release.py`共89项通过。
- 发布测试采用先失败后通过，覆盖RX-V2开关、7个显式loss权重、P1～P5嵌套矩阵、GPU映射、单一V和评估白名单。
- 既有真实checkpoint无query smoke：PASS；`query_inputs=0`、`target_inputs=0`、两次fresh teacher前向和一次memory视图。
- 本地Git Bash探针：FAILED。系统将指定Git Bash错误路由到失效WSL；未执行后续本地shell payload。远端原生Bash语法检查与dry-run为发布前检查。
- N607直连preflight：VERIFIED。普通账号、项目根目录与8张RTX3090可见。
- 初始GPU快照：GPU1约3538MiB、GPU7约3528MiB；其他卡约6477～11570MiB。发布前将再次读取GPU和进程归属，不触碰既有任务。

## 4.发布参数

- 本地工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-daot-stn-v1`
- 远端Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 基础checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 远端release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_daot_stn_rx_v2_p1_p5_manysig_s392005_20260903_r1`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_daot_stn_rx_v2_p1_p5_manysig_s392005_20260903_r1`
- launcher：`code/scripts/launch_phase1_adv3b02_daot_stn_rx_v2_p1_p5_manysig_s392005_20260903.sh`
- 启动环境/CWD：release内`CVS-RFFI-repo`，使用上述Conda Python；dispatcher通过nohup脱离。
- 启动命令：在release代码根执行launcher默认5行矩阵。

## 5.停止规则与预期产物

仅在数据/query边界错误、错误checkout或run root、输出覆盖、launcher-wide故障、至少两行出现相同确定性异常、无prediction闭合或scorer连接错误时停止；不得因中间性能低而停止，也不得触碰无关进程。

每行预期至少生成：`final_ssdg.pth`、`metrics_joint.json`、clean与3个LEO_WEAK逐场景结果、per-RX拆分结果、训练日志、状态文件和阶段2原型导出。5行全部闭合后才可标记`ARTIFACTS_COMPLETE`。

## 6.发布与启动证据

- 2026-09-03 17:49 CST直连preflight再次通过；数据、基础checkpoint可见；release归档、release目录和run root均确认不存在。
- release归档：`phase1_adv3b02_daot_stn_rx_v2_p1_p5_manysig_s392005_20260903_r1.zip`；本地与远端唯一SHA256均为`4f765d0f85cb5a2fa7664edc7aa9157ad9deba62a09c8edfae820590c52d66a2`。
- 远端Python compileall与两份launcher的`bash -n`均通过。
- 远端dry-run实际生成5条`RXV2-ROW`、5条训练命令和5条评估命令；首次验收因错误搜索不存在的`MUSE-DRY-RUN`标记而返回非零，读取完整输出后确认执行成功，并用真实稳定标记重新核验为`VERIFIED`，没有修改冻结代码或参数。
- 2026-09-03 17:51 CST启动；dispatcher锚点PID=`826695`，CWD为release内`CVS-RFFI-repo`。
- 启动后核验：5个worker、5个训练主链及其数据加载子进程存在；P1/P2/P3/P4/P5分别绑定GPU1/7/0/2/4；5份dispatcher日志与5份train.log均生成并增长。
- 启动后GPU占用：GPU0/1/2/4/7约5353/5179/9740/9660/5369MiB；已有任务未被停止或修改。
- 启动日志中的协议读回一致：`L/U/V=6300/56700/27000`、`legacy_l_u_v`、130+70=200epoch、seed392005；异常扫描为0个`Traceback/CUDA out of memory/RuntimeError/Killed`。
- 当前尚无完整epoch指标和最终评估，不得形成性能或晋级结论。
