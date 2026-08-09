# P1-CP-SFCE冻结设计与实现追溯卡（2026-08-09）

状态：`LOCAL_VERIFIED_INDEPENDENT_REVIEW_ALLOW`。本卡只覆盖source-only的P1-CP-SFCE六折C/G续训实现；不构成性能、晋级或Phase3 unknown能力结论。

|ID|冻结要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|
|CPSFCE-01|C保持GeoSat-C；G保留CB-SFCE同一LEO focal CE、`lambda=.10`、`gamma=1`、round-robin与local4×3数据合同|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local|数值与C恒等测试|不得改变loss、采样、head或optimizer|
|CPSFCE-02|仅在G上以base梯度投影新增SFCE梯度；无eps，`b=0`合法|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local|投影单测|`dot<0`时精确去除冲突分量|
|CPSFCE-03|AMP使用scaled辅助VJP、captured scale除回raw；base一次backward/unscale；统一clip/step|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local|AMP等价训练环测试|禁止未投影SFCE第二次backward|
|CPSFCE-04|全trainable VJP与encoder/head作用域审计；None、断图、非有限或scope外非零均fail-closed|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local|正负测试|精确head为`id_backbone.cls_head.head`|
|CPSFCE-05|step-state增量、首epoch技术marker、local4×3覆盖及每scope至少一次冲突终态合同|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local|receipt/terminal测试|四项逐batch计数必须相等且>0；不读取性能决定训练|
|CPSFCE-06|原子且best-effort的失败收据不遮蔽原异常|`phase1_cp_sfce.py`,`train_ssdg.py`|verified-local|writer失败测试|不写raw或样本数据|
|CPSFCE-07|固定12臂、八卡≤2、40epoch、final-only、严格warm-start|launcher、测试|verified-local|`bash -n`与dry-run|新根不可覆盖|
|CPSFCE-08|本地语法、focused、CB必要回归与真实lite_d无query冒烟|测试与本卡|verified-local|已执行|不访问N607|

## 冻结机制

设共同GeoSat-C损失为`L_base`，按当前batch出现TX等权的卫星focal CE为`L_sfce`，其中`lambda=.10`、`gamma=1`。C仅优化原有`L_base`；G对LEO logits路径上的共享encoder和精确分类head参数作用域`S`，以`b=∇L_base`、`a=.10∇L_sfce`构造：当`||b||^2>0`且`a·b<0`时，`a'=a-(a·b/||b||^2)b`，否则`a'=a`。随后该作用域写入`b+a'`；作用域外只保留共同base梯度。该机制不是表示、特征或logit teacher对齐。

## 预期验证

已在`ssr-gpu`完成：`py_compile`；CP-SFCE focused 6项；CP-SFCE+CB-SFCE必要回归17项；CPU等价缩放训练环中的lite_d无query forward/backward；`bash -n`；12条dry-run；`git diff --check`。结果均通过，只有既有AMP弃用警告。独立复核结论为`P0=0，P1=0，ALLOW`。

追溯汇总：verified-local=8，deferred=0，rejected=0，blocked=0。
