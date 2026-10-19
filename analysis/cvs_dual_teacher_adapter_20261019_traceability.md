# 双教师＋浅中层时序残差适配器：实现追踪

来源：用户新增附件 adf48c3d-6ad2-43ef-88b1-01357f6738dc/已粘贴的文本.txt，及本chat确认学生追加10轮、累计E30。目标保持low urban>60%、clean相对匹配NoAs可重复正收益、逐suburban差距；新用户明确授权新增非分类头特征模块和首轮冻结F/defaultW，覆盖旧仅原F正常联合训练的本轮限制。

本表服务实现范围与结果交付，属NONBLOCKING记录；不增加设计审批、report/hash、重复review或数据重验门槛。实现、六行固定E30训练与七视图独立评分均完成；固定源资格FAIL，方案有效性未成立，整体目标未完成。

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|T01|§4/9|固定合法Tc与Ts、source-first选择；不按target择师|teacher.py / design.py|verified|已有CSIL來源inspection/payload與inputJacobian同IQV360；Tc为固定seed B0 FINAL20|Ts远端可用性、实际严格加载和全训练均已验证；E200成本单列，不冒充E20|
|T02|§1/9|复用已完成训练/同L新channel/V诊断|ER与D3 final_analysis.md|verified|D3 actual1441NN，sameorigin180/different0，SOURCE_FAIL固定|旧epoch-first关闭，不救援|
|T03|§6|t1/t2原GN之前并联浅中层时序残差|adapters.py|verified|实际native block窄测，新增4768参数；全六行真实父CP smoke、strict FINAL30恢复及七视图已完成|rank16，k5 DW/GELU，维持原pool/tap|
|T04|§6|仅B输出层0初始化；无双零gate|adapters.py / checks_adapters.py|verified|11负例，train/eval同RNG相等，B首步及后续A/D有效梯度，strict reload|A/D正常初始化，首步A/D梯度0正常|
|T05|§6|原F与默认head冻结，梯度仍穿过后续层|model.py / adapters.py|verified|native block＋固定后层/合成6×160头回传PASS；实际F/CP冻结与梯度路径接入、全六行训练和预测已完成|新用户明确覆盖旧默认联合训练限制；不是另训头|
|T06|§6/9|新checkpoint严格恢复adapter与累计继承状态|model.py|verified|先strict父F再装adapter；eval先装结构再strict新CP|不可套用scratch20/ancestors[]旧guard|
|T07|§4/7|Tc和学生用同clean视图，Ts和学生用同practical IQ|teacher.py / model.py|verified|实际同tensor/生成标识和class order witness|两教师eval/stopgrad；不混源概率路由|
|T08|§7|全L真实CE；仅teacher错误时减弱KD，不看student置信|distillation.py|verified|全128 CE、wrongteacher mask、KD归一化和梯度窄测|无branch alignment/HQ/MixStyle/Fishr/新伪标签|
|T09|§7|显式处理六类顺序/输出尺度/温度|teacher.py / design.py|verified|从固定来源读取并记录logit scale及固定温度规则|不用target校准；不raw-logit MSE|
|T10|§9|S0/S1/S2/S3统一head/data/初始化/预算|design.py / source.py / runner.py|verified|学生E20+10累计E30，用户已明确确认|匹配NoAs及教师shadow成本控制在最小矩阵中披露|
|T11|§9|保留原数据/U隐藏真值/测试七视图truth-last|source.py / evaluate.py / runner.py|verified|复用L6300/U56700/V27000与现有received IQ|不改SNR/场景/物理ID/单包推理|
|T12|§6/9|分别报告参数/反传/teacher前向/信道/墙钟|observer.py / report.py|verified|实际计数与原F冻结/默认头/J160更新可达性|冻结不宣称廉价或clean必保持|
|T13|§9|源资格先固定；完整性能与配对误差/多seed口径|qualification.py / report.py|verified|先源规则后formal，无失败近邻救援|n1仅探索，不宣称可重复|
|T14|§5/9|标准浅中层LoRA仅在主S3有效后独立比较|后续独立实验|deferred|用户明确条件性后续|首轮不同时展开多个adapter族|
|T15|§8|raw IQ补充分支仅在局部修正不足后独立比较|后续独立实验|deferred|用户明确条件性后续|不做IQ MSE去噪器，不先新增分支|
|T16|§7.3|clean特征keep约束仅KD仍明显漂移后考虑|后续独立消融|deferred|用户明确必要时，不首轮叠加|不用epsilon0限制未知RX改错空间|
|T18|§6.3|恢复原默认头联合更新仅在后续独立比较|后续独立实验|deferred|用户要求先统一冻结 W 验证特征|本轮不改头自由度|
|T17|项目最小流程|一次实际code/config提交、窄负测/CPsmoke、P0/1、archive/compile、live证据|runner / 最小prereg|verified|按项目唯一whitelist完成，不增seals/hash/审批|本表为NONBLOCKING记录，不新增release gate|

固定现有教师：Tc=原B0纯F seed2026092701 FINAL20（不按目标最高分选择）；Ts=CSIL csil-s2026092701固定last200，scratch/9800updates/合法相同物理角色。既有同IQ源V360每项比较：d0mid44修复/18新增错，d1mid65/11，d0low72/21，d1low76/16；总净13.263889pp。该证据不是完整V1800，也不是新独立blind test。旧nontrueKD和inputJacobian方向固定配方已关闭；本轮改变的是监督分工与新增特征承载空间，不把旧教师识别优势等同于新学生有效。

首轮结构限定：rank16、k5、bias=False；两处adapter分别约2000和2768参数，共4768；原900722参数F保持完整。具体计数须实际模块读回，不用估计替代。不是标准线性LoRA，不能声称可折叠成原单权重矩阵。标准LoRA/残差适配/蒸馏的研究依据分别见 [LoRA](https://arxiv.org/abs/2106.09685)、[Residual adapters](https://arxiv.org/abs/1705.08045)、[Knowledge distillation](https://arxiv.org/abs/1503.02531)，无线任务收益仍需本实验验证。

实际接入约束：原native CENTER_UNITS optimizer要求Sinc和W可训练，不能直接拿来给冻结F/adapter使用。保留CoordinateSinc参数化与已加载函数，适配阶段Sinc冻结/更新0，显式仅adapter优化器；S0若更新骨干则另作统一冻结head的实际优化器规则，不能为绕过断言开启W。

实际闭环：r01在首更新前教师构建重入，已保留FAILED/0update；r02 code/config 9a376ab313b3ba42459df32694c1502675e02a59，六行全2220steps/累计6660、默认head冻结、adapter原F冻结、同L/U顺序、teacher/As完整计数、strict FINAL30加载与all7 truth-last完成。源决策先固定FAIL，fullVclean相对较强NoAs −0.903704pp；正式S3 clean82.348810%、lowurban42.326190%。不延长/扩种子/扫系数；source-only下一问题另存当前run。详细数值见 automation_reports/CV-SincNet/20261019-phase1-F-dualTeacher-adapter-e20plus10-s2026092701-m6-r02/final_analysis.md。

## 追加：冻结源拟合与joint160诊断

- Run `20261019-diagnostic-F-dualAdapter-sourceFit-e30-s2026092701-m8-r01`：ANALYZED，9361身份前向/900原物理信道包/180合法L，0optimizer/EMA/head/V/target更新或新前向（V/target为0）。
- S3末次已见118/180，Ts154/180；low同场景已见S3为28/59、Ts42/59。clean真实joint160中位漂移S3=.242783、S1=.111271；零输出eval一致为0，Dropout-only T4²KL=.885004而720/720全对。
- 用户条件“局部修正不足后再考虑rawIQ补充编码器”已有源证据支持；仅完成4256参数/356352卷积MAC/当前输入同IQ的可行性核对，NOT_IMPLEMENTED/NOT_LAUNCHED。LoRA的S3有效前置条件未成立，本轮不混featurekeep。
- 完整结果：[final_analysis.md](../automation_reports/CV-SincNet/20261019-diagnostic-F-dualAdapter-sourceFit-e30-s2026092701-m8-r01/final_analysis.md)。旧配方SOURCE_FAIL保持，未扩seed/E50或新target读出。

## raw IQ 补充路径实现（新候选）

- Run `20261019-phase1-F-rawIQ-dualTeacher-e20plus10-s2026092701-m4-r01`：独立新root，4新row（NOAS_ADAPTER/S1/S2/S3），相同合法B0E20+10累计E30；旧6row产物复用，不加载旧失败S3权重。
- 用户条件raw路径：20261019-diagnostic-F-dualAdapter-sourceFit-e30-s2026092701-m8-r01的同IQ源证据已满足局部拟合不足/Ts额外知识前置；仅改变特征输入通路，未重扫权重/温度/seed。
- 已实现当前received2×256→两级16通道卷积GN/GELU→两个零B投影，在t1/t2原GN前注入；4256参数/8新键/卷积MAC356352，原206状态与defaultcos30头保留。无跨forwardclean缓存、oracle部署输入、新头或旧localadapter叠加。
- 实际F合成验证零输出joint/logits逐值相同，原状态保持，两个B有梯度；本地aggregate68负例与源决定/元数据10负例通过，0真实CP/IQ/remote。真实CPsmoke为launch首步，PASS继续。
- 训练raw/As正确数取已有logits，标注stochastic读出；全4固定FINAL30自动7view预测，源3条件固定后独立truthlast。标准LoRA/featurekeep暂不混入。
- 当前状态：RUNNING，真实父CP smoke PASS后三个worker首轮健康完成，S3排队；运行commit6ea2f71f6，尚未获得本轮最终性能。最小预登记：[report.md](../automation_reports/CV-SincNet/20261019-phase1-F-rawIQ-dualTeacher-e20plus10-s2026092701-m4-r01/report.md)。
