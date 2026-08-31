# WISER-RF P3-Primary退化梯度修复pilot预登记

## 状态与修复边界

- run ID：`wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1`；当前状态：`RUNNING`；Git提交：`fe9ec1424e8396d1f6a4e8931a8653750cbb74e9`。
- 本提交仅将精确D92的退化方差在`sqrt`前安全屏蔽，并为零`delta`预构造安全分母；不改变正常坐标的前向公式、P3方法、训练预算、arm、scene或晋级门槛。
- 两个前序run分别因历史绑定漂移和退化坐标非有限梯度在无query smoke阶段停止，均为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，未进入pilot、未打开query、未产生prediction，原run/log根永久保留。
- 回归证据：`zero_identity/zero_fft/ill_conditioned`三类D92反向测试先失败后通过；10个D92/P3/pilot/scoring/Target25测试文件完整通过；相关模块`py_compile`通过。

## 冻结协议、输入与矩阵

- protocol=`p2_min_v1`，data status=`VALIDATED_ONCE`；outer=`rx_3_19__seed_713102__k_10__new_5`，receiver=`3-19`，seed=`713102`，K=`10`，new-count=`5`。
- capsule=`d92-e0-full-target125:5910674066e8bbf93684fddd6af6fd2cef7e8f208d64e403ac7e58030a2a8cc5`；split=`d92-e0-full-target125:rx_3_19__seed_713102__k_10__new_5`。
- arm=`N0,N1,N2,N3,N4,N5,N6`；scene=`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；共21个独立prediction/receipt。训练、选择和插值固定`query_rows_used=0`，全部support状态冻结后只读打开query，prediction完整后由独立scorer连接truth。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- manifest：`/home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json`；source summary：`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component/int8_domain_class_prototypes.npz`。
- P3 config=`configs/wiser_rf_p3_primary_20260831.json`；source binding=`configs/wiser_rf_adv3b02_source_binding.json`。

## release、资源、路径与命令

- release=`wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1_fe9ec142.tar.gz`；远端根=`/home/szu2070436088/2510044040/CV-SincNet/releases/wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1_fe9ec142`。归档大小36272459字节，本地/远端唯一SHA256均为`0199fe792f5c38f2b0dfc010fc1d0a4c14117dfb935346c641fec26dd2d6a931`；远端11个相关模块一次编译通过。
- run root=`/home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1`；log=`/home/szu2070436088/2510044040/CV-SincNet/logs/wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1/pilot.out`；score=`<run-root>/score`。四个新目标均已确认不存在。
- pilot为单进程顺序工作流，冻结物理GPU0，`CUDA_VISIBLE_DEVICES=0`后程序使用`cuda:0`；启动前再次盘点且每GPU不超过用户授权的3个训练实验。

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_wiser_pilot.py p3-smoke --manifest <manifest> --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260831.json --checkpoint <checkpoint> --source-summary <source-summary> --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root <run-root>/smoke --device cuda:0 --runtime-commit fe9ec1424e8396d1f6a4e8931a8653750cbb74e9 --arm N6 --scenario leo_clear_weak
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_wiser_pilot.py p3-pilot --manifest <manifest> --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260831.json --checkpoint <checkpoint> --source-summary <source-summary> --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root <run-root>/pilot --device cuda:0 --runtime-commit fe9ec1424e8396d1f6a4e8931a8653750cbb74e9 --arms N0 N1 N2 N3 N4 N5 N6
```

## 停止、artifact与晋级

- 仅因协议/query/truth泄漏、错误split/receiver/seed/K/scene、输出冲突、可微与精确D92不同构、非有限loss/gradient、prediction不完整、scorer绑定错误、进程归属不清或确定性重复异常停止；不得因低性能停止。
- 预期artifact：smoke结果、21个support audit与prediction/receipt、completion marker、独立详细score、资源记录和`pilot_auto_result.json`。
- pilot门槛不变：P3 BA三scene中位提升≥3pp、最差scene≥-0.5pp、P3 floor中位及low-elev floor不下降、P1/P2每scene≥-2pp、zero-id=0、条件数≤基线2倍、至少2/3scene净help为正；N1不得成为冠军。
- 仅当三个scene完整且`full_target25_authorized=true`才发布Target25；否则报告科学未晋级。Target25通过才授权K10扩展，Stage B不在本run自动执行范围内。

## 启动核验

- 远端owner PID=`2958724`（PPID1），控制PID=`2958726`，当前smoke worker PID=`2958727`；worker CWD精确指向本run的`fe9ec142`release，cmdline为预登记`p3-smoke`，输出根为本run的`smoke`。
- worker映射物理GPU0 UUID=`GPU-56adac86-77cd-36c9-8770-dbf002650461`，启动采样显存7646MiB；GPU0启动前已有2个训练进程，加入本run后总数为3，未超过用户授权上限。
- 本地启动SSH在取得PID后因远端owner保持连接而主动断开；远端owner已脱离为PPID1且继续存活。首次日志采样为0字节，属于stdout缓冲，进程状态为运行且GPU已建立compute context；后续只读检查日志/artifact增长，不重复启动。
