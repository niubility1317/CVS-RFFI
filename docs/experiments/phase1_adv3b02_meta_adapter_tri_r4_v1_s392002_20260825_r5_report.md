# CVS_META_ADAPTER_TRI_R4_V1 r5最小预登记报告

- run ID：`phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5`
- 状态：`ARTIFACTS_COMPLETE`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 固定代码与配置提交：`8d07f752e5093766f31edab7fdc97159c60d70f1`

## 候选与矩阵

P0为冻结base控制；P1为随机adapter；P2为source监督adapter；P3为FOMAML固定LR；P4为FOMAML+Meta-SGD。P1～P4顺序运行，科学失败不阻断后续候选。Phase1只使用source角色训练和选择，最终checkpoint分别评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。r5与r4配置除不可覆盖run ID外完全相同；r4因包含已知Torch／NumPy原型写入缺陷而保持`NOT_LAUNCHED`。

## 本地验证与定点P0/P1审查

- RED测试模拟`Tensor.numpy()`返回不兼容数组，旧实现稳定在冻结原型artifact路径失败；GREEN实现仅对6类冻结原型使用有界列表桥接，NPZ数组由当前NumPy创建。
- checkpoint bundle继续保存原Torch张量，训练、adapter、source选择、四场景评价和Phase2 query路径均未改变。
- Phase1真实入口整文件36项通过；Meta-Adapter Phase1／Phase2邻近宽回归228项通过。N607同一`CVS-RFFI`环境内存复现确认桥接后数组类型身份正确，NPZ写入和无pickle回读成功。
- 定点P0/P1审查范围仅为原失败路径及其直接消费者：未发现会导致run启动错误、输出覆盖、协议越权、缺少合法prediction或错误修改checkpoint原型类型的问题。

## N607最小预登记

- N607账户：普通`N607`用户`szu2070436088`
- 环境：现有`CVS-RFFI`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：0；启动前再次核对占用，且不得超过每GPU两个训练进程。
- release归档本地路径：`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5_a465e329.tar.gz`
- release归档远端路径：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5_a465e329.tar.gz`
- 本地release归档固定提交为`a465e3292a2c4c60069eb8d4dbffb685fcb2b709`，35491571字节、5007个条目，包含r5配置和报告；SHA256=`8becfa7e4a8e68aa3bae1c8668c81b2b8f6bb6af47617c0cc1b9349203e2c349`。
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/checkout`
- 冻结checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5.out`
- 启动命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py --config configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5 --base-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --gpu 0`
- expected artifacts：每个P1～P4子目录的`logs.jsonl`、`metrics.csv`、`selected_meta_bundle.pt`、`source_adaptation_curve.json`、`run_summary.json`、`p0_control_evaluation.json`、`final_checkpoint_evaluation.json`、`frozen_prototypes.npz`，以及矩阵级`candidate_matrix_summary.json`。
- 技术停止规则：仅在协议越权、错误checkout/output root、输出覆盖、无法产生规定artifact、launcher-wide故障，或至少两个候选出现相同确定性pre-prediction异常时停止；不得因低准确率停止。

## 当前边界

- r3已封为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；r4未同步、未启动且不再使用。
- r5的P1～P4均已自然完成，矩阵级汇总选择P4并给出`SOURCE_SELECTION_ELIGIBLE`。该结论只表示Phase1 source-held-out选择完成，不代表Stage2-B目标域适配已有正向收益。

## N607发布证据

- 发布前核对r5 release目录、run root和stdout日志均不存在；r3／r4／r5无活动进程，GPU0～GPU7均无计算进程，项目盘剩余7.3TiB。
- release归档仅同步一次；远端SHA256=`8becfa7e4a8e68aa3bae1c8668c81b2b8f6bb6af47617c0cc1b9349203e2c349`，与本地一致。
- 远端checkout中的7个相关生产入口编译通过；真实绝对checkpoint、ManySig、r5配置和output root的launcher dry-run通过，且dry-run前后均未创建run root。
- 发布与远端落地已完成；后续启动健康检查和完整Phase1产物闭合见下文。

## N607启动健康检查

- 2026-08-25 05:38（Asia/Hong_Kong）完成唯一一次不可覆盖启动；launcher PID=`2650209`，P1训练子PID=`2650534`。
- launcher和训练子进程的CWD均为预登记r5 checkout；cmdline分别精确绑定r5配置、run root、冻结checkpoint、ManySig、P1配置和GPU0。
- stdout从不存在增长到576字节，P1已写入`_configs/P1.json`和`P1/config_snapshot.json`；异常扫描计数为0。
- GPU0只出现P1子进程，初始显存594MiB；15秒复核时子进程保持`R`状态，累计CPU时间38秒并持续推进。
- 当前最高状态提升为`RUNNING`。该证据只证明发布和启动健康，不代表P1完成、source选择通过或存在目标域正向收益。

## Phase1完整闭合

- launcher PID=`2650209`及P1～P4候选子进程均自然退出，未执行kill、restart或覆盖操作；完成后GPU0～GPU7均无计算进程，stdout异常扫描计数为0。
- 矩阵级`candidate_matrix_summary.json`状态为`ARTIFACTS_COMPLETE`；P1～P4均生成非空`run_summary.json`、`selected_meta_bundle.pt`、`frozen_prototypes.npz`、source适配曲线、P0控制评价和最终checkpoint评价。
- r3曾失败的原型写入点在P1真实生产运行中通过：`frozen_prototypes.npz`为4412字节，可无pickle回读；P4的`selected_meta_bundle.pt`为4309666字节。
- 严格加载P4 bundle通过：总参数1058341，实际可训练参数8670，占比0.8192%，仅覆盖`id_backbone`／`dom_backbone`的time、freq和fusion adapter，共30个非分类参数名；正式适配步数为3，符合≤1%和≤40步预算。

## Phase1结果

下表为各候选最终checkpoint相对同候选P0控制的变化，单位为百分点；每格依次为旧类均值变化／旧类floor变化。

|候选|clean|leo_clear_weak|leo_low_elev_weak|leo_rain_weak|
|---|---:|---:|---:|---:|
|P1随机adapter|-0.0167／-0.1571|-0.0214／-0.2429|+0.0143／-0.1571|-0.0226／-0.1857|
|P2 source监督|-0.6131／-3.7071|-0.0310／-3.7786|-0.0952／-4.5143|-0.2048／-4.8357|
|P3 FOMAML固定LR|-0.8881／-5.5714|-0.3429／-2.8214|-0.5298／-2.7786|-0.4655／-2.2500|
|P4 FOMAML+Meta-SGD|-0.8833／-5.5643|-0.3369／-2.8071|-0.5262／-2.7571|-0.4655／-2.2429|

- P1的两个source-held-out适配准确率分别为1.0000和0.8333；P2～P4分别为1.0000和0.9167，最差source退化均为0。
- P2、P3与P4的source-held-out选择值相同；P4以221.16ms的较低选择延迟胜出，因此矩阵选择P4。
- clean和三类LEO结果显示P4最终checkpoint相对其P0均有负向变化，尤其floor退化明显。这里不能宣称域适应正收益；是否晋级只由下一阶段同row Target5的`DA1_REG0-DA0_REG0`旧类均值和floor门槛决定。

## 下一步

使用P4 bundle和冻结原型构建同row单seed Target5最小矩阵；Phase2仅消费`p2_min_v1`、`VALIDATED_ONCE`固定LEO received IQ和合法target support标签。先完成真实checkpoint无query smoke，再产生truth-free prediction，最后由独立scorer连接truth。Target5旧类均值至少+1.0pp且floor至少+0.5pp才进入Target25，否则记录科学失败并推进下一少层候选。
