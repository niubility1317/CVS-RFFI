# P1-CP-SFCE冻结设计与实现追溯卡（2026-08-09）

状态：`LOCAL_V2_VERIFIED_INDEPENDENT_REVIEW_ALLOW`。v1在C侧出现默认遥测未初始化、在G侧于首次或后续AMP尺度下报告base参数梯度非有限；该技术停止没有性能结果。v2只修复AMP分类与C控制臂遥测，使用新不可覆盖run ID`phase1_cp_sfce12_20260809_v2`；独立复核结论为`P0=0，P1=0，ALLOW`。本卡仍只覆盖source-only的P1-CP-SFCE六折C/G续训实现，不构成性能、晋级或Phase3 unknown能力结论。

|ID|冻结要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|
|CPSFCE-01|C保持GeoSat-C；G保留CB-SFCE同一LEO focal CE、`lambda=.10`、`gamma=1`、round-robin与local4×3数据合同|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local|数值与C恒等测试|不得改变loss、采样、head或optimizer|
|CPSFCE-02|仅在G上以base梯度投影新增SFCE梯度；无eps，`b=0`合法|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local|投影单测|`dot<0`时精确去除冲突分量|
|CPSFCE-03|AMP使用scaled辅助VJP、captured scale除回raw；base一次backward/unscale；统一clip/step|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local-v2|真实CUDA GradScaler base/aux伪溢出及lite_d无query训练环|禁止未投影SFCE第二次backward|
|CPSFCE-04|全trainable VJP与encoder/head作用域审计；None、断图、非有限或scope外非零均fail-closed|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local-v2|raw VJP正负测试|精确head为`id_backbone.cls_head.head`|
|CPSFCE-05|step-state增量、首epoch技术marker、local4×3 applied覆盖及每scope至少一次冲突终态合同|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local-v2|receipt/terminal测试|`attempted=applied+base_skips+aux_skips`；不读取性能决定训练|
|CPSFCE-06|原子且best-effort的失败收据不遮蔽原异常|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local|writer失败测试|不写raw或样本数据|
|CPSFCE-07|固定12臂、八卡≤2、40epoch、final-only、严格warm-start|launcher、测试|verified-local|`bash -n`与dry-run|新根不可覆盖|
|CPSFCE-08|本地语法、focused、CB必要回归与真实lite_d无query冒烟|测试与本卡|verified-local|已执行|不访问N607|

## 冻结机制

设共同GeoSat-C损失为`L_base`，按当前batch出现TX等权的卫星focal CE为`L_sfce`，其中`lambda=.10`、`gamma=1`。C仅优化原有`L_base`；G对LEO logits路径上的共享encoder和精确分类head参数作用域`S`，以`b=∇L_base`、`a=.10∇L_sfce`构造：当`||b||^2>0`且`a·b<0`时，`a'=a-(a·b/||b||^2)b`，否则`a'=a`。随后该作用域写入`b+a'`；作用域外只保留共同base梯度。该机制不是表示、特征或logit teacher对齐。

## v2 AMP技术合同

若base已scaled backward并unscale后的任一参数梯度非有限，则一次性对全部trainable参数计算未缩放`L_base` VJP；若`S`内存在None、任一已连接raw梯度非有限（包括scope外），立即fail-closed。仅raw全有限时，调用GradScaler公开skip/backoff路径：不投影、不重试、不更新optimizer、EMA或prototype，且记录`BASE_SCALED_OVERFLOW_RAW_FINITE`、触发参数名、pre/post scale与state未推进证据。辅助scaled VJP非有限时，保留图后一次追加raw`.10L_sfce`VJP；raw非有限、`S`断图或scope外非零都fail-closed；raw全有限则同样整batch跳过并记录`AUX_SCALED_OVERFLOW_RAW_FINITE`。没有“连续第几次overflow”规则。

C每batch初始化CP遥测为空字典，冻结终态固定为`CONTROL_ARM_NOT_APPLICABLE`。G终态要求`attempted=applied+base_skips+aux_skips`、projection/outside-audit/optimizer-state-step均等于`applied>0`、所有raw故障/no-step为0、每次skip均raw-finite且scale严格下降/state不推进；local4×3每格必须有applied rows/loss/finite/nonzero证据，双scope冲突计数仅按applied批次并各至少一次。

## 预期验证

v2本地已完成：`py_compile`；CP-SFCE focused 8项；CP-SFCE+CB-SFCE必要回归19项；真实CUDA GradScaler的base/aux伪溢出回退与raw非有限/断图/scope外负测；真实lite_d无query forward/backward；C控制臂终态遥测测试；`bash -n`；12条dry-run；`git diff --check`。独立复核实际运行上述验证并给出`P0=0，P1=0，ALLOW`；既有AMP弃用警告不影响本次结果。

追溯汇总：verified-local-v2=8，deferred=0，rejected=0，blocked=0。

## CP-SFCE postfreeze追溯（INDEPENDENT_REVIEW_ALLOW）

本节只定义训练技术完整后的final-only证据闭环；不读取性能来选择训练、阈值或checkpoint，也不构成Phase3 unknown能力声明。

|ID|来源要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|
|CPSFCE-PF-01|12个clean开发导出、12个source-only三LEO场景导出、12个source-proxy连续诊断、6个CPU串行C/G pair，共42步|`launch_phase1_cp_sfce_postfreeze_20260809.sh`|verified-local|`bash -n`、dry-run42|训练root固定`phase1_cp_sfce12_20260809_v2`，新postfreeze根不可覆盖|
|CPSFCE-PF-02|严格checkpoint/NPZ SHA、strict-load、`dual_cvsincnet_tx_logits_v1`分类头合同、local4类序、C/G逐行metadata/physical/scenario绑定|`eval_phase1_cp_sfce_pair.py`、测试|verified-local|head-contract/manifest-path/C-G swap负测|不加载checkpoint权重，不拟合/校准/选参|
|CPSFCE-PF-03|runtime view=`single`与manifest satellite profile、三场景物理互斥、每场景完整TX/RX|evaluator、测试|verified-local|payload负测|source-only LEO，固定seed和场景|
|CPSFCE-PF-04|clean四floor、每LEO场景四floor、fold三场景等权overall、全局18格等权overall及proxy非补偿门|evaluator、测试|verified-local|six-fold aggregate测试|proxy仅guardrail，不能补偿任何分类门|
|CPSFCE-PF-05|prior JSON只能来自同matrix/root、固定v2 training root、冻结pair/source-TX/C-G候选路径；technical严格true、proxy finite[0,1]，F6仅聚合F1--F5|evaluator、测试|verified-local|cross-root/v1-root/pair/source/arm-swap/technical/proxy负测|输出必须final-only且拒绝覆盖|

本地与独立验证：`python -m py_compile`通过；CP与未修改CB postfreeze focused合计`38 passed`；`bash -n`通过；dry-run精确为12个clean＋12个LEO＋12个proxy＋6个pair；`git diff --check`通过（仅CRLF提示）。独立复核结论`P0=0，P1=0，ALLOW`；不发布、不访问N607。
