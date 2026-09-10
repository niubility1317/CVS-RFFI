# CORE90交叉响应基线恢复审计

日期：2026-09-11。范围：配置、矩阵、历史损失隔离和局部回归。未启动正式训练；本文件不证明E200性能或全训练轨迹复现。

## 核对依据

- 历史`E:/type10-7/code/snapshots/phase1_adv3_mechanism32_queue_20260701/train_ssdg.py`、`losses.py`和`launch_phase1_adv3_mechanism32_queue_20260701.sh`。
- [CORE90技术报告](../docs/CVS_ADV3B02_QKNNV42_TECHNICAL_REPORT_20260709.md)、[本次实施计划](ADV3B02_CORE90_CROSS_RESPONSE_IMPLEMENTATION_PLAN_20260911.md)。
- 当前`code/SSDG/train_ssdg.py`、`code/cvsrffi/losses.py`、`code/model_dual_cvsincnet.py`和`code/post_stage_cli.py`。

历史快照未独立保存模型及公共数据默认文件，不能称为历史逐位复现。当前模型谱系与报告核对为lite_d、身份feat_joint、域feat_imp、160维、共享sinc/hf、身份no_dac、域no_stats、rcn_stats强度0.35。模型内部完整数值回归由集成测试承担。

## 固定的历史有效配置

[配置文件](../code/configs/phase1_core90_cross_response_v1.json)以历史parser字面默认值为底，再解析launcher的`set_candidate_defaults`与B02分支覆盖值，不依赖当前训练入口的隐式默认值。另明确公共数据/批量值和当前协议覆盖。公共数据文件未存在于旧快照中，对应值是本次匹配预算决定，不冒充旧文件逐位证据。

|项目|有效设置|
|---|---|
|训练日程|E200，label130+pseudo70，AdamW相关lr=0.0002、weight_decay=0.0001，label_smoothing=0.01|
|基本损失|domain=1，adv=0.35，orth=0.05，cons=0.08，group_ce=0.16，FISHR=0.04，lambda_u=0.16，entropy=0.01|
|EMA/伪标签|EMA=true、decay=0.999，tau_min=0.92、tau_max=0.97、quantile=0.86；原门控保留|
|原型|lambda_proto=0.0032、domain_align=0.10、margin=0.15、push=0.10、min_count=2|
|身份紧致|lambda=0.032，E8起、25epoch warmup，supcon/radius/CVaR=0.30/0.35/0.35|
|开放空间几何|lambda=0.0024，E12起、25epoch warmup，tail=0.14、vacuum=0.40|
|B02 proxy unknown|lambda=0.0045，E45起，hard48，core_q=0.90、accept_q=0.85、CVaR_alpha=0.30、core_accept_weight=0.45|
|soft unknown mixup|lambda=0.0045，E25起，count24、order3、alpha0.5，CE=0.60、vacuum=0.35，detach=false|
|source episode|lambda=0.0035，E20起，radius_cap=33度、mixup=0.75，保留三西格玛半径|
|LEO|clean/LEO拼接、卫星仅TX CE；lambda_sat_cls=0.68、lambda_sat_cons=0，E80起；三种leo_*_weak|
|LEO视图日程|`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|匹配预算|clean batch_size=128；P4×Q4×K2为32条独立物理记录的块；不能把每个batch默改成32|

## 允许且必须披露的协议变化

历史launcher使用.10/.70/.20、joint_safe和周期test评价。本次固定source-only的L_s/U_s/V=.07/.63/.30，`from_scratch=true`、`baseline_ckpt=''`、`checkpoint_selection=final_only`、E200最终模型。`best_metric=clean_val_tx`仅作为源验证遥测，joint_safe=false；历史test选模和Phase2导出均关闭。当前source-only训练入口才是禁止test接触的执行保证，不能仅凭`test_eval_start_epoch=201`视为完成协议隔离。

显式使用三角色`legacy_l_u_v`代表本次计划中的L_s/U_s/V，不继承当前其他方法的V_cal/V_select四角色比例。source_cal_ratio/source_select_ratio均0。源/目标物理RX和train/test day列表必须由调用者显式提供；矩阵不复制旧默认RX划分。RX列表必须非空、无重复、非负且source/target互斥。跨day是否有实际holdout以输入列表为准，不凭命名声称未见day泛化。

没有加载任何初始化、teacher或旧选模权重。历史方法结构继承不等于checkpoint数据契约兼容。矩阵只生成argv/resolved配置，不调用训练、SSH或GPU调度。

## 已发现并修复的损失语义漂移

对历史与当前顶层函数做AST比较：45个旧损失函数/类中38个未变、7个变化；没有缺失函数。变化包括`_safe_angle_from_cos`、`open_world_feature_space_loss`（返回标注）、`proxy_unknown_energy_loss`、两个虚拟负样本helper、`source_episode_three_sigma_loss`和`PrototypeMemoryBank`。

|变化|对匹配基线的影响|处理|
|---|---|---|
|proxy component radius变成默认core_quantile=.80|原B02为robust three-sigma；即使不启用新loss也改变判定几何|显式three_sigma；配置路径绑定历史实现|
|source episode默认min_three_sigma_core|改变跨域角度阈值与梯度|显式three_sigma；绑定历史实现|
|PrototypeMemoryBank domain/push改为当前batch可微中心|旧实现使用memory原型，历史部分项无当前特征梯度；同权重不等于同方法|保留历史loss/update，仅增加状态序列化|
|虚拟bridge/helper、数值角度处理扩展|hard48生成/梯度不能凭同CLI判等|整个历史loss文件原样隔离，内部helper也保持历史版本|

[legacy_losses.py](../code/cvsrffi/cross_response/legacy_losses.py)从旧`losses.py`按字节复制；[baseline_compat.py](../code/cvsrffi/cross_response/baseline_compat.py)提供局部训练绑定表，不能全局替换其他方法。U0同样必须绑定历史loss。适配器只接收列举的后续API参数：新增机制权重必须0，半径必须three_sigma；未知参数或启用的新机制立即报错。失活机制的有限数值阈值允许传入，但不会影响历史数学。

当前新增的顶层lambda_*全部显式设0，MUSE、CRRA、sat_anchor_adapter关闭；不叠加A1、DAOT或其他后续机制。仅恢复旧方法不修正其既存科学不足；原型项的历史无梯度语义也不能悄悄改进后称为U0。

补充AMP数值修复：真实CUDA短测中，U0和U4_bilinear均连续16步梯度非有限，scaler从65536回退到1仍未更新。异常定位到历史proxy-unknown的余弦矩阵乘法及后续角度链路：在autocast中，即使输入`.float()`，矩阵运算仍会降至半精度，边界夹紧值可能舍入至1。适配器现将历史损失代数放入FP32区间，半精度输入先提升，再执行原文件中的公式；模型前向仍可AMP。该修改同样用于U0和全部变体，不改变公式、权重、采样或历史原型梯度语义。原历史文件继续按字节保留；FP32历史loss/gradient/单步回归仍逐值一致。实际AMP修复后结果见[实现验证报告](CORE90_CROSS_RESPONSE_IMPLEMENTATION_VERIFICATION_20260911.md)。

## 矩阵和验证

[矩阵工具](../code/scripts/core90_cross_response_matrix.py)登记U0、U1、U2、U3、U4_additive、U4_bilinear、U5、Ux、head_only、permanent_detach。U1—U4使用相同P/Q/K、结构seed、角色轮换、隔离和基础batch。U5相对U4_bilinear仅启用反馈；Ux不混入响应/决策损失。head_only只学习响应辅助头，permanent_detach保留真实决策损失但永久切断响应到身份路径。每个variant/model_seed输出目录独立，重复seed/variant拒绝；写矩阵文件采用排他创建，已有文件不覆盖。

15项测试已通过：矩阵实际开关和输出独立、输入协议负测、未知参数和混杂机制负测、proxy/source损失与梯度及一次SGD更新逐值匹配、历史原型两步loss/梯度匹配和状态序列化，以及全部10个variant的真实训练parser解析。两个新CLI参数接入后重跑完整测试文件，15项通过，耗时7.54秒。

测试入口：[test_cross_response_matrix.py](../code/tests/test_cross_response_matrix.py)。这组测试不证明完整optimizer/EMA/采样序列一致、不证明模型架构历史逐位一致，也不证明GPU训练或E200科学性能。
