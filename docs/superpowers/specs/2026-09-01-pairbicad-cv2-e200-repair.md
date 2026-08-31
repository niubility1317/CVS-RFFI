# PairBiCAD-CV2-E200修复版冻结说明

## 1.目标

本修复版替代旧run`phase1_pairbicad_cv2_screen24_seed392002_20260831_r1`。旧run因用户要求从固定update/coverage停止改为完整200epochs而停止，既有完整和partial artifact仅保留作历史证据，不参与新run选模或续训。

新run的12个候选、fold1/fold8、seed392002、day1/day2/day3、source-only边界和`concat_sat_ce_only+LEO_WEAK`协议保持不变。所有候选必须从头训练完整200epochs；`optimizer_update`只作为遥测横轴，不再作为训练终止条件，科学收敛状态只记录、不提前停止。

## 2.十一项修复

1. strict Pair候选每个物理样本必须形成真实LEO视图，禁止`clean_duplicate`；三种`LEO_WEAK`从训练早期全部可达。
2. 统一所有候选为200epochs，删除6500updates、coverage周期和24小时作为正常终止预算。
3. 正式训练接入真实`CoverageLedger`：U累计/唯一物理样本覆盖、L端TX×receiver×day最小暴露和完整分组暴露均写入遥测。
4. 实现coverage warmup，然后才由source-only`V_cal`驱动`ReduceLROnPlateau(factor=0.3,patience=3,min_lr=1e-6)`。
5. `no_early_freeze`必须成为可执行约束；启用时200epochs内相关训练参数不得提前冻结，并记录审计。
6. `adversarial_two_time_scale`必须成为显式运行分支；开启时判别器与encoder使用分离优化器，`LR_D/LR_encoder=1.5`。
7. T1/T3测量pair梯度相对TX任务梯度，并把有效比例限制在5%以内。
8. T2/T3把困难组30%质量上限真正接入Margin-REx/CVaR权重，不得只记录常量。
9. D3动态GRL同时消费判别器准确率、TX margin、对抗梯度比和冲突信号；conditional与`z_dom`剂量独立有界。
10. `V_cal`只用于scheduler/状态；`V_select`仅在final/EMA/SWAD候选形成后执行一次选择，两者物理样本不交叉。
11. 增加EMA候选，与final/SWAD一起在冻结的`V_select`上一次比较；选择结果不得反馈训练、调参或重跑。

## 3.矩阵与资源

- 候选：`CV2-B0/B1/B2/B3/D0/D1/D2/D3/T0/T1/T2/T3`。
- fold：1和8；seed：392002。
- 训练天：day1/day2/day3。
- 每行：200epochs，从头训练。
- GPU：0—7，每张卡最多2个本run主训练进程；资源不足时排队，不改变矩阵。
- B3、D0和T0是三个静态分支基线ID，但功能配置允许相同，不称为12个唯一功能配置。

## 4.完成定义

每行必须记录`epoch=200`、final/EMA/SWAD候选身份、一次`V_select`选择、CoverageLedger、机制审计、严格checkpoint重建、Clean和三种`leo_*_weak`评估以及`ARTIFACTS_COMPLETE.json`。低性能是科学结果，不得触发停止或重跑。

本说明不启用Phase2、目标接收机、support、query或truth，也不恢复报告中明确延期的VICReg、pair-delta、soft-U CDAN、sparse XDC、Fishr、FastTrust或HCF transport。
