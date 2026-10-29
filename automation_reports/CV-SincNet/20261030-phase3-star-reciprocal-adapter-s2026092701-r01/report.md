# PAMF Phase3 U池互近邻适应与两阶段匿名注册

## 预登记（2026-10-30）

Run `20261030-phase3-star-reciprocal-adapter-s2026092701-r01`，实际code `d3d8f5680fd72dc7ded272ffd1ded3143e490244`；唯一主reciprocal::DA1_REG1，self_pl为匹配控制，两臂都完整DA0/1×REG0/1。详细机制/参数/角色和停止规则见`docs/PAMF_PHASE3_STAR_RECIPROCAL_ADAPTER_PLAN_20261030.md`、对应config；矩阵唯一维护在experiment.json。

原F0 Tri-SIG-TIME SSL E50/11100/default6head/232993与DURA R69/266951、oldneighborhood源聚合prior(codecb58217de28eb2cb5536d2392aef268b79006877)和pairedMLP6τ全部冻结，不继承上轮relationOE失败F，不回读source IQ/逐样本features，不调readout/τ。sourceL6300/Us56700/V27000及OE成员监督为祖先，目标标签0不等于零源监督。真实head输入梯度及原logits、F/G参数冻结的合成G10checkpoint smoke首步、在U首次读取前执行。

合法G10 RX24-1 U384/2021-03-08用于适应/发现/注册，futureQ864/2021-03-15只逐样本预测；source RX1/3/4/6/8互斥。使用已有capsule/split/VALIDATED_ONCE，不重抽raw记录或信道，P0就是星地received、P1为同received的冻结G输出。receiver2027/scene2026103001/channel2026104001及原sixscene概率保持。算法无targetTX/真K/配额/Qtruth；正式RX24-18不读，已暴露开发RX/设备/Q保持开发身份。

两臂小g=160→32ReLU→160/10432参数、CPU seed2026092701同init、final0；unit空间预归一化残差L2≤.1，归一化后恢复原raw范数，原default6头不替换/重训。原U margin≥0且MSP.95为trusted；selfPL仅这些自身argmax，主对accepted nontrusted中≥3个trusted MNN10邻居且modalvote≥.8扩标签，其他U不强归旧六。FP64余弦/opaque-ID稳定ties；两臂全部pseudo在任何optimizer前固定。每臂fullU200步AdamW .001/wd.0001/clip5，CE伪标签+allU offdiagGram1，无增强/采样/EMA/早停；空mask显式0step且U/Q完全旁路g保持精确DA0，不以舍入差伪造适应收益。

DA0/DA1各U-only注册：两old margin均负的候选在当前mutual10NN图取连通分量，K可不同或0；min16、与trusted邻接点比例≤.05、交替两半中心距≤LOO95%半径；只导出center160/半径，不在Q回读U featurebank。REG先保留oldmargin≥0的原旧label，仅old拒识后允许anonymous ball，防旧类被匿名竞争覆盖。该性质不证明未来新类有效或later拒识，需独立Q。

全部g/registry/τ冻结先Q；共享一次U和逐批Q的原F0 P0/P1提取，两adapter顺序独立。全部2×4prediction完成后CPU scorer先核publicgate/finitebudget/jointoldgate/ID和任意K，再读取Utruth冻结四个独立匿名映射，全部map完成才Qtruth；trueK/映射不回算法。AN分母含全部真实到达设备，漏发现0；REG0 AN/HOW N/A，未匹配anonymous是acceptedwrong非reject，缺必需TX/scene覆盖N/A。

主报宏AK/RU/H，附CK/原同输入F0closed损、AN/laterRU/同DA REG旧保留/HOW、逐sceneTXRXday/CI及配对得失和U伪标签/聚类污染诊断。DA1REG0须绝对CK90/RU90/F0loss2pp及positiveoldAKgain，主DA1REG1须AN80/intendedCK90/laterRU90/oldprotect2pp/REGretain2pp且七主组覆盖；DA0未适应对照的绝对门槛只诊断，不额外要求其先达90才验证改进。source90不前置，不以control替主、不自动扩seed/formal。开发PASS仅下一独立确认资格，完整goal仍ACTIVE/finalfalse。

### 执行、存储、资源及成本

- N607 Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-reciprocal-adapter-s2026092701-r01-release`，唯一owner `codex-phase3-star-reciprocal-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_reciprocal_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-reciprocal-adapter-s2026092701-r01-release/runtime.json`。
- 不可覆盖 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-reciprocal-adapter-s2026092701-r01`，predict config13公有字段，无truth/K/targets。单GPU predict/adapterworker内部两臂全部四态后独立CPUscore；truth和targets仅后者argv。actualfree≥4096MB卡0..7取最高，≤2新增own/GPU，本次1；foreign不停止/修改/重启。
- 源/目标F0和G只读，预期realF0full2×1248=2496+合成4，G1248+合成2，headgradient合成2另计。若两mask非空：g训练153600行+终态诊断768行，U/Q额外g/head各2496行；空mask按实际0更新/旁路，计数在summary/costs。原source/Gcache/Phase1祖先成本保留，不当独立新样本。
- 实际sourcecontainer存/每次evidence算四个3233读出，不报成只一份；F/G/四readouts512876参数，单adapter523308、同时双adapter533740，非NNgeometry/anchors/registries另计。每realrow baseline+两个adapted evidence共3次（空mask减相应次数），共14976 realMLP行forward与smoke8；无B1实测，不以批量耗时或显存采样预报加速。
- 预期launch/pipeline/configs/logs；predictsmoke、两adapter model/200steps/summary、U_evidence/pseudo/每armDAregistry/diag、Ufreezecomplete、all8predictions/complete；score scores/complete及observations。权重/IQ/逐行pred/rawlogs private，不提交Git。
- 技术fail保留partial，科学负分/emptyseed/K0不重启/重选τ/改K；UNKNOWN先核原PID/root，不重复提交或触碰foreign。

验证：集成48 passed, 1584 warnings, 39 subtests passed in 11.10s、10AST/实际runtimePASS；之后仅producer4定向PASS5.15s含1新增empty-mask精确旁路（与主suite3项重叠，不累加为独立全suite）。唯一独立直接P0/P1 PASS。一次preflight source原F0/聚合/公有capsule/split/G在位、未读UQ或truth值，8卡free11195–18793MB。单code/configcommit、单archiveSHA比较/remotecompile、真实checkpoint首步smoke、实际PID/CWD/argv/GPU/log增长及prediction全完后truthlast；不增签名/报告审批/重复数据验证。
