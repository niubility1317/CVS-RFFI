# PairBiCAD P0–P4正式矩阵报告

## 当前状态

- 状态：`LOCAL_VERIFIED_PENDING_IMPLEMENTATION`。
- Run ID：`phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r1`。
- 方法范围：`P0/P1/P2/P3/P4`；P5–P9不属于本run。
- source数据：`Dataset_WigSig/ManySig.pkl`；训练day1/day2/day3。
- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- folds：1、8；seeds：392001、392002、392003；总计30行。
- 训练预算：每行4000 optimizer updates；物理batch48，clean/LEO拼接网络batch96。
- GPU：N607 GPU0–7；每GPU最多2个本run训练进程。
- 目标、Phase2、support、query、truth：禁止访问。
- 每行预期artifact：final checkpoint、runtime/audit、metrics、clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`、资源遥测、`ARTIFACTS_COMPLETE`或技术失败marker。
- 允许技术停止：错误candidate/fold/receiver/day/seed/update、target/Phase2越权、输出覆盖、错误release/CWD、命令无法运行、无合法checkpoint/四场景闭合、同一确定性pre-prediction异常至少2行或进程归属不清。
- 低性能不得停止、重启、热补丁或选择性重跑。
- 兼容性裁决：旧D0–F3保持`concat_sat_ce_only/E80/0.68/0`；新P0–P4预登记`ce_only_plus_pair_selfsup`，卫星TX标签仍只走CE，但P3/P4允许无标签pair/VICReg/delta自监督。该候选级覆盖来自当前报告实现授权，不改变`LEO_WEAK`、source-only或L/U信息权限。

## 报告到实现追踪

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|PB-P0-01|第3、15、16节|每update clean/LEO严格成对、单次拼接前向|config、trainer、train_ssdg|pending|聚焦单前向测试|物理48/网络96|
|PB-P0-02|第5节|有标签clean/LEO TX CE，卫星权重0.5→1.0|trainer、config|pending|loss调用审计|保持concat_sat_ce_only标签边界|
|PB-P1-01|第4、10节|z_dom因素化为z_r/z_d/z_c/z_int|heads、trainer|pending|维度和隔离测试|z_int=24|
|PB-P1-02|第14节|shared-stem gradient firewall|trainer、gradients|pending|firewall应用计数|scale=0.05|
|PB-P2-01|第5节|有标签类条件receiver/day/channel DANN|heads、trainer|pending|CAdv调用和梯度测试|不使用U硬标签|
|PB-P2-02|第7节|z_dom主因素反向TX对抗，排除z_int|heads、trainer|pending|TXAdv输入隔离测试|仅L|
|PB-P3-01|第8节|L/U pair identity hinge|pair、trainer|pending|有效数与有限值|epsilon=0.05|
|PB-P3-02|第8节|U clean/LEO预测JS|pair、trainer|pending|无U标签测试|不产生伪标签|
|PB-P3-03|第8节|projector VICReg防坍塌|pair、trainer|pending|variance/covariance测试|projector=128|
|PB-P4-01|第9节|identity delta信道对抗|pair、trainer|pending|delta adversary审计|L+U|
|PB-P4-02|第9、10节|domain delta解释信道，rx/day稳定|pair、trainer|pending|channel预测与pair损失|L+U|
|PB-P4-03|第9节|delta norm hinge|pair、trainer|pending|半径测试|delta=0.25|
|PB-GRAD-01|第14节|动态对抗梯度剂量与日志|gradients、trainer|pending|rho审计|adv0.15–0.25|
|PB-PROTO-01|第2节、项目协议|L/U信息权限和source-only|train_ssdg、tests|pending|协议负测|禁止target/query|
|PB-COMPAT-01|新报告与旧默认冲突|P0–P4候选级扩展，旧候选不变|config、train_ssdg、runtime|pending|旧候选回归+新runtime测试|不得把新模式冒充旧默认|
|PB-MATRIX-01|第21、22节|P0–P4×2fold×3seed完整矩阵|launcher、shell|pending|30行dry-run|U4000|
|PB-EVAL-01|指标章节、AGENTS|clean+三种LEO_WEAK逐行闭合|launcher|pending|artifact测试|不能只报aggregate|
|PB-DEFER-01|最终推荐|P5 soft-U CDAN|无|deferred|N/A|P4多seed source证据后再议|
|PB-DEFER-02|最终推荐|P6 XDC、P7 margin-tail|无|deferred|N/A|不混入P4首轮|
|PB-DEFER-03|第17、18节|P8 hard-LEO、P9 SWAD、Fishr/MixUp|无|deferred|N/A|后续独立消融|

## 版本与发布

- Git分支：`codex/phase1-pairbicad-p4-20260831`。
- 设计：`docs/superpowers/specs/2026-08-31-pairbicad-p4-design.md`。
- 实现计划：`docs/superpowers/plans/2026-08-31-pairbicad-p4.md`。
- code/config commit：pending。
- N607 release与命令：pending。
