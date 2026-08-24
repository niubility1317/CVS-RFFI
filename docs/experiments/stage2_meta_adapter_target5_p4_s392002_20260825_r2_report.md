# CVS_META_ADAPTER_TRI_R4_V1 P4 Target5 r2最小预登记报告

- run ID：`stage2_meta_adapter_target5_p4_s392002_20260825_r2`
- 状态：`ARTIFACTS_COMPLETE`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 修复代码提交：`6b66fce0c1e3a4a9315e539999fbe62f0798881e`
- 固定计划提交：`c489dc8df100ea6c7cd79ad135f9a0f07725d2d0`

## 恢复边界

- r1的Target工厂已完整生成15个truth-free row，但真实checkpoint无query smoke在NumPy2.2.5／Torch2.1.0 ABI桥接处失败；r1已封为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，prediction矩阵从未启动。
- r2只修复Stage2的NumPy／Torch输入输出桥接，不改变P4 bundle、冻结原型、receiver、operating point、seed、场景、support／query物理样本、capsule、split、3步适配或判决规则。
- 复用r1已完成的工厂输出`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_p4_s392002_20260825_r1/matrix_config.json`。数据仍为同一`p2_min_v1`、`VALIDATED_ONCE`切片，不因代码桥接或run ID变化重验。

## 候选与矩阵

- 候选：P4 FOMAML+Meta-SGD；单seed：`392002`；Target5 receiver：`20-1`。
- operating point：`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`；每点三类LEO weak，共15个row。
- Phase2仅读取固定received IQ、合法target support标签、P4 bundle和冻结原型；不读取source／clean样本、source cache、query真值或query角色。query不得更新任何状态。
- 原编码器内可训练参数8670／1058341，占比0.8192%；正式更新3步；无D92式协方差、LDA或新增／持久分类头。

## 本地验证与定点复审

- RED测试同时屏蔽`torch.from_numpy`和`Tensor.numpy`，旧路径在IQ转换处稳定复现N607错误。
- GREEN实现使用`torch.frombuffer`转换IQ、整数标签和冻结原型；prediction写盘通过Python值生成当前NumPy数组，避免后续同一ABI故障。
- 69项Stage2工厂／runner／matrix／handoff／scorer／row export回归通过；199项Meta-Adapter Phase1／Phase2邻近回归通过；相关文件编译通过。
- 修复后的唯一一次定点复审未发现P0/P1：桥接函数不接收source、clean、query truth／role，不改变适配顺序、参数选择、步数、冻结状态或判决规则；prediction写盘仍在DA0／DA1推理完成后发生。

## N607执行预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- 新release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_p4_s392002_20260825_r2/checkout`
- release归档：`E:\type10-7\release_archives\stage2_meta_adapter_target5_p4_s392002_20260825_r2_d81564fd.tar.gz`；固定提交`d81564fdac933f051704f9f33d8315566507df1b`，35500199字节、5010个条目，SHA256=`890927e0815b1893f5fee933c58729d6bbdc403f7272555854c2a8e48cfb044e`。
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_p4_s392002_20260825_r2/stage2_meta_adapter_target5_p4_s392002_20260825_r2_d81564fd.tar.gz`
- smoke output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_p4_s392002_20260825_r2_smoke`
- prediction output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_p4_s392002_20260825_r2`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_p4_s392002_20260825_r2.out`
- expected artifacts：`smoke_receipt.json`；每row的`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`和`receipt.json`；truth-last `score.json`；矩阵级`target5_summary.json`。
- 技术停止规则：仅在协议越权、query或source泄漏、错误checkout／row／split、输出覆盖、prediction不完整、scorer连接错误、launcher-wide故障，或至少两个row出现同一确定性pre-prediction异常时停止；不得因低性能停止。

## 科学晋级规则

prediction完整后才由独立scorer连接truth。15个同row score聚合`DA1_REG0-DA0_REG0`：旧类均值至少+1.0pp且旧类floor至少+0.5pp才晋级Target25；否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`并推进下一少层候选。

## N607发布与smoke证据

- 启动前核对新release root、r2 smoke root、prediction root和stdout日志均不存在；GPU0～GPU7均无计算进程，项目盘剩余7.3TiB。
- release归档仅同步一次；远端SHA256=`890927e0815b1893f5fee933c58729d6bbdc403f7272555854c2a8e48cfb044e`，与本地一致；8个相关生产入口远端编译通过。
- N607现有`CVS-RFFI`环境未安装pytest，未改动环境或安装包；运行时验证由真实checkpoint smoke承担。
- P4真实checkpoint无query smoke通过：`status=REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS`、`checkpoint_load_strict=true`、`backward_count=3`、`trainable_fraction=0.008192066640147174`、`query_opened=false`、`source_opened=false`、`query_state_update_count=0`，且`performance_result=null`。
- 当前最高状态为`LANDED`；下一步立即启动唯一一次15-row truth-free prediction矩阵并执行PID／CWD／cmdline／GPU／日志增长健康检查。

## Prediction闭合与scorer定点修复

- 唯一一次15-row矩阵自然完成，矩阵receipt为`PREDICTIONS_COMPLETE`，15／15 row均生成非空`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`和`receipt.json`；矩阵级`truth_opened=false`、`source_opened=false`，完成后无活动进程或GPU残留。
- 首次truth-last scorer在第一条new10 row拒绝连接并且未写出`score.json`。只读诊断确认当前场景prediction与truth各320个opaque ID，两个集合完全相等；truth侧文件共含三场景960行。
- 根因是scorer把“含数值`true_class_index`的CVS多场景sidecar”误判为无场景简表，错误使用三场景全集做join。该问题只在独立评分阶段，不影响已经冻结的prediction、适配状态或query边界。
- RED测试复现多场景数值truth误连接；GREEN实现先识别CVS场景sidecar，再按receipt场景完成全量opaque-ID join，并且仅对该场景的`target_old`计算REG0指标。scorer全测18项及Stage2工厂／runner／matrix／handoff／scorer／row export整组70项通过。
- 当前最高状态为`ARTIFACTS_COMPLETE`；15-row prediction不重跑。发布独立scorer修复后，从尚未产生score的truth-last阶段继续。
- scorer修复归档：`E:\type10-7\release_archives\stage2_meta_adapter_target5_p4_s392002_20260825_r2_scorerfix1_cef0ee30.tar.gz`；固定提交`cef0ee30a998a3f2acfcf52c257edb0d19f1e575`，35502573字节，SHA256=`467aef9c963b3842f3e5ccf89258fa8c4d0d198dd85f66b4a5e8fb687d02fc78`。
- scorer修复远端release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_p4_s392002_20260825_r2_scorerfix1`。
