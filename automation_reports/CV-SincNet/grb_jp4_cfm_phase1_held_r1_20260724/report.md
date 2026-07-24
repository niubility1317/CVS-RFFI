# GRB-JP4-CFM-qKNN-D92/r2-sharedK1 Phase1留出反证预注册

根目录权威报告：`E:\type10-7\automation_reports\CV-SincNet\grb_jp4_cfm_phase1_held_r1_20260724\report.md`。

当前状态为`LOCAL_FEASIBILITY_REJECTED / NOT_LANDED / NO_PERFORMANCE_RESULT / TARGET25_CLOSED`。候选把support-only rank4更新合入真实`joint_proj.0.weight`，K1只由6个target-old与ground锚共享识别，新类单例只注册；K5/K10增加old/new各0.5的严格physical-LOO margin监督。冻结设计审查曾为`P0=0、P1=0、P2=0`，但这只批准本地实现。严格LOO与完整qKNN资源合账后，K10的`M_DA92=284,775B`，超过262,144B硬门22,631B，且support MAC收据漏算outer-fold两次完整refit。held固定54行和Target25均未运行。

既有证据同排结论：D62和SVRN均已完成完整125；D92完成完整125；D91仅完成固定`K10/new5,receiver=20-1,seed=713101`的15条outer row，且15/15预测与D62逐值一致，不能冒充125。D62五切片总体为`B/A/Floor/N/H/F=81.51/64.39/35.15/59.11/61.09/17.11`；D92按五个等量25行切片重算为`81.55/65.56/36.81/58.93/61.57/15.99`；SVRN为`73.10/43.03/11.21/23.46/29.25/30.07`。D92具有旧类/floor/遗忘正信号但新类略退，D62注册后大多数状态回退而几乎等于D81，SVRN全面弱于D62；三者均未达绝对目标。完整逐切片、D91边界、SCXMAP/ADV3B02代理和资源表见根目录权威报告。

设计阶段第二次快速复审曾为`P0=0、P1=1、P2=3`；随后把第二轮重线性化固定为增量累加`theta^(t+1)=Pi(theta^(t)+g_K*u^(t+1))`，并闭合normal equation符号、rank仅记录、coverage零分母及状态同步，设计冻结终审为`P0=0、P1=0、P2=0`。最终实现/release审查的`P1=7`是后续独立结论，优先于设计冻结结论。

最终快速终审`P0=0、P1=0、P2=0`，批准进入本地实现；不授权N607或Target25。

实现前发现旧4,096B子门与最坏18个ground多原型的实际payload冲突。冻结后资源契约勘误已把该子门限定为`update-factor wire`；ground wire另行完整计入`M_DA/M_DA92`各自262,144B总state。实现必须分别报告`update_factor_wire_bytes/ground_wire_bytes/total_component_bytes/full_arm_state_bytes`。独立勘误审查=`P0=0、P1=0、P2=0`，科学方法、held门和Target门不变。修订后冻结文档SHA256=`616e3bccdfc8ec97de213caa747faa9110e7a5b07c77040d09b1e92d07f54a2c`。

实现追踪最终为verified=6、blocked=6、rejected=4。旧`68 passed`和真实checkpoint smoke在严格physical-LOO修正后已失效；当前排除Held evaluator的窄验证为`60 passed`，独立复核为基础组件`40 passed`、Stage2组件`18 passed`，Held falsifier因资源gate为`9 ERROR`，没有性能结果。最终独立release审查为`P0=0、P1=7、P2=0`：除资源超限和support MAC漏算外，ground-off正式API、D92 fold常量自证、row级执行/调度、外部COMMIT sidecar以及公开协议承载均未闭合。代码已作为被否决研究原型保存在本地提交`1ed87d66`，不授权N607，不同步公开协议，不访问Target25。完整审查与资源表见根目录权威报告。
