# CVS阶段性成果技术报告：`ADV3B02_CORE90_SOFT_E200`与`qKNNV42`

日期：2026-07-09
作者：Codex本地审计与四子agent交叉验证
承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`
本地协议源：`E:\type10-7\AGENTS.md`、`E:\type10-7\项目.md`

## 1.阶段性结论与声明边界

本报告把`ADV3B02_CORE90_SOFT_E200`和`qKNNV42`定义为当前CVS路线的阶段性成果，但不把它们写成真实在轨部署完成或Phase3 unknown/open-set完成。

阶段性结论如下：

|对象|当前可声明结论|不能声明|
|---|---|---|
|`ADV3B02_CORE90_SOFT_E200`|Phase1 source-only弱标注/半监督跨接收机DG表征基座；在ManySig源域上形成较强`z_id`身份表征、`z_dom`域表征、proxy/virtual unknown边界治理和prototype导出资产|不能声明Stage2 old/new适应已由它单独完成；不能声明真实unknown_FAR/FPR95改善；不能声明真实卫星IQ或在轨部署成功|
|`qKNNV42`|冻结ADV3B02特征上的Phase2轻量support-memory注册/适应头；主线是同一目标接收机LEO目标域内target-old适应+target-new seen-new注册识别|不能写成新训练backbone；不能写成端到端深度模型；不能把历史字段`target_unknown`当真实unknown拒识成功；不能把Phase3诊断负证据包装成成功|
|当前组合成果|在N20 HP08L5注册新类包上存在同row阶段性候选：`old_acc=94.52%`、`min_old=85.71%`、`seen_new_acc=90.14%`、`min_new=81.43%`、`H_old_new=92.28%`|该行没有独立真实unknown集合；unknown/FAR不属于此行主结论|

协议边界来自`项目.md`：Phase1是source-only地面弱标注/半监督DG；Phase2主线是Stage2-B target-old适应和Stage2-C old+seen-new注册；unknown/open-set是Phase3备用或诊断项。所有结果解释必须保留`R_s/R_t`、`Y_old/Y_new/Y_unknown`和K-shot边界，禁止把不同候选的单项最值拼成一个结果。

## 2.本轮证据来源与多agent交叉验证

本报告基于本地只读审计、现有报告和代码路径；未访问N607远端、未启动训练、未修改实验脚本。

四个子agent分工如下：

|子agent|审计重点|交叉验证结论|
|---|---|---|
|ADV3B02机制审计|模型结构、训练入口、loss、参数、B02机制和Phase1结果|确认`ADV3B02_CORE90_SOFT_E200`是Phase1 source-only候选，机制为`core90/accept85/alpha30`，不是Stage2/unknown完成证据|
|qKNNV42机制审计|support-memory结构、量化、打分、辅助视图门控、相似方法|确认`qKNNV42`是冻结特征上的压缩KNN/prototype部署头，不是独立训练模型|
|结果证据审计|同row指标、artifact路径、unknown诊断负证据|确认当前可用阶段成果是K5 strict N20 HP08L5注册新类行；unknown/FAR仍是负证据或未纳入主线|
|协议与报告结构审计|`项目.md`边界、章节口径、禁写项|确认最终报告必须写成CVS协议约束下的阶段性技术报告，而非完整在轨系统报告|

本地Git状态边界：`E:\type10-7`根目录不是Git仓库；本报告写入Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`。承载面已有与本任务无关的未提交改动，本报告只新增本文档。

## 3.CVS科学场景与符号定义

CVS面向天基RFFI的核心问题是：在地面源接收机上获得低标签率训练数据后，学习能够迁移到目标接收机/目标LEO链路的发射机身份表征，并在目标接收机域内用少量K-shot support完成旧类适应和新类注册。

符号定义：

|符号|含义|
|---|---|
|`x`|原始IQ片段或由IQ派生的目标域特征输入|
|`y`|发射机身份标签，即TX类|
|`d`|域标签，通常由receiver/day/rx_day/channel/satellite view等组成|
|`R_s`|源接收机集合，用于Phase1训练|
|`R_t`|目标接收机集合，用于Phase2部署/评估，必须与`R_s`不相交|
|`Y_old`|源域已知旧TX类，也是目标域需要保留识别能力的旧类|
|`Y_new`|目标域注册新TX类，Phase2 Stage2-C可用少量support学习|
|`Y_unknown`|真实未知TX类，Phase3/diagnostic-only，不能倒灌到Phase2主线|
|`z_id`|身份表征，用于TX分类、prototype、few-shot support memory和qKNN检索|
|`z_dom`|域/信道/接收机扰动表征，用于吸收receiver/day/channel/satellite nuisance和域诊断|

CVS阶段边界：

|阶段|目标|允许使用的数据|主指标|
|---|---|---|---|
|Phase1|source-only弱标注/半监督DG表征学习|`R_s`上的ManySig旧类、低标签`L_s`、无标签`U_s`、源域派生LEO压力视图|source闭集DG、strict UDU、receiver floor、satellite floor、proxy/virtual unknown风险、prototype导出质量|
|Stage2-B|目标旧类少样本适应|`R_t,Y_old`的K-shot support和query|`old_acc`、`min_old_class_acc`|
|Stage2-C|目标旧类适应+seen-new注册|`R_t,Y_old`和`R_t,Y_new`的K-shot support/query|`old_acc`、`seen_new_acc`、`min_old`、`min_new`、`H_old_new`|
|Phase3|未知类拒识备用/诊断|互斥`Y_unknown`只作eval-only，除非另有显式Phase3协议|AUROC、FPR95、unknown_reject、FAR、known保留率|

本报告主结论只对应Phase1和Phase2 Stage2-C no-unknown主线。所有unknown/FAR证据单独列为负证据。

## 4.CVS总体方法流水线

CVS方法可以写成三层：

1.源域表征层：`raw IQ->CV-SincNet/CVS->z_id,z_dom`。`z_id`承载TX身份几何，`z_dom`承载receiver/day/channel/LEO扰动。

2.源域训练层：在`rho_label=0.1`低标签比例下，把`L_s`监督CE、`U_s`伪标签SSL、域监督/域对抗、source episode、proxy/virtual unknown边界塑形和源域LEO压力视图合成一个source-only训练目标。

3.部署注册层：冻结Phase1 backbone，把目标接收机LEO域的K-shot support压缩为int8 support memory、class prototype、scenario metadata和少量校准标量。`qKNNV42`不重训backbone，只在冻结`z_id`空间进行少样本检索、旧类锚定和新类注册仲裁。

## 5.`ADV3B02_CORE90_SOFT_E200`方法细节

### 5.1身份与实验入口

`ADV3B02_CORE90_SOFT_E200`是ADV3机制32候选矩阵中的B02项。候选摘要为`core90/accept85/alpha30`，目标是提高known core保真并降低proxy/virtual unknown接受风险。

|项|值|
|---|---|
|候选ID|`ADV3B02_CORE90_SOFT_E200`|
|机制摘要|`direct vaccept core90 accept85 alpha30`、known core保真更强|
|训练入口|`E:\type10-7\code\SSDG\train_ssdg.py`|
|启动脚本快照|`E:\type10-7\code\snapshots\phase1_adv3_mechanism32_queue_20260701\launch_phase1_adv3_mechanism32_queue_20260701.sh`|
|数据|`Dataset_WigSig/ManySig.pkl`|
|split|`tx_rx_day_1_7_2`|
|标签比例|`labeled/unlabeled/source-val=0.10/0.70/0.20`|
|训练轮数|`epochs=200`|
|最佳epoch|`best_epoch=194`|
|label阶段|`label_epochs=130`|
|pseudo阶段|`pseudo_epochs=70`|
|seed|`392002`|
|checkpoint|`best_joint_safe_ssdg.pth`|
|prototype导出|`phase2_zid_prototypes.json/.pt`|

本候选的后续引用路径包括`E:\type10-7\code\scripts\launch_phase2_adv3b02_frozen_manytx_unknown_diag_20260706.sh`，该脚本把B02的`best_joint_safe_ssdg.pth`作为冻结基模型checkpoint。

### 5.2模型结构：物理先验CV-SincNet+双分支解耦

ADV3B02训练入口构建`DualCVSincNetDisentangle`，不是单一路径普通CNN。其核心是两个CVSincNet分支：

|分支|输出|功能|
|---|---|---|
|identity backbone|`z_id`、`tx_logits`|学习TX身份不变特征，用于闭集分类、prototype、Phase2 few-shot和qKNN|
|domain backbone|`z_dom`、`dom_logits`|学习receiver/day/channel/LEO扰动特征，用于域监督和域解释|
|GRL adversarial head|`adv_dom_logits`|对`z_id`做域对抗，减少身份表征泄漏接收机/信道信息|
|TX adversarial head|`tx_adv_logits`|约束`z_dom`不要承载过多TX身份信息|

底层`CVSincNet`包含以下物理分支：

|组件|作用|
|---|---|
|Sinc滤波器前端|以可解释带通滤波捕获IQ频带结构|
|time branch|从时间序列提取局部包络、相位和瞬态模式|
|DAC branch|使用widely-linear complex block建模IQ非圆性、DAC不平衡和硬件缺陷|
|frequency branch|提取频域/滤波组统计特征|
|PA branch|使用memory polynomial lift和包络门控建模PA非线性与记忆效应|
|CosFace/物理感知分类头|用角度margin增强身份类间分离|
|`feat_id/feat_dac/feat_pa/feat_imp/feat_joint`|保留身份、DAC、PA、重要性和融合特征，供后续诊断与导出|

代码证据：`E:\type10-7\code\model.py`中`class CVSincNet`、`sinc_out=48`、`sinc_kernel=79`、`emb_dim=256`、`freq_bands=64`、`pa_memory_depth=4`、`margin_s=30.0`；`E:\type10-7\code\model_dual_cvsincnet.py`中`DualCVSincNetDisentangle.forward`返回`z_id`、`z_dom`、`dom_logits`、`adv_dom_logits`。

### 5.3训练机制总览

ADV3B02不是只靠一个CE loss训练。其训练分为label阶段和pseudo阶段：

|阶段|数据|主要动作|
|---|---|---|
|label阶段|源域标注样本`L_s`|监督TX分类、域监督、域对抗、几何约束、proxy/soft未知压力、source episode、LEO压力视图|
|pseudo阶段|源域无标签样本`U_s`+标注样本|继续监督训练，同时用EMA/当前模型为`U_s`生成TX伪标签，经多重门控后做strong view CE和熵约束|

伪标签只来自source unlabeled，不是target receiver伪标签，也不是unknown类伪标签。伪标签模块不能被写成Stage2 target-new学习或Phase3 unknown拒识。

### 5.4训练损失函数与优化目标

ADV3B02总loss可概括为：

```text
L_total =
  L_tx
+ λ_dom L_dom
+ λ_adv L_adv
+ λ_group L_group_ce
+ λ_fishr L_fishr
+ λ_proto L_proto
+ λ_zid L_zid_compact
+ λ_ow L_open_world_feature
+ λ_proxy L_proxy_unknown
+ λ_soft L_soft_unknown_mixup
+ λ_src L_source_episode
+ λ_u L_pseudo_ce
+ λ_ent L_entropy
+ λ_sat_cls L_satellite_ce
+ λ_sat_cons L_satellite_consistency
```

各项含义如下：

|loss|作用|ADV3B02中的边界|
|---|---|---|
|`L_tx`|源域标注TX分类CE，训练`z_id`身份识别能力|闭集身份主目标|
|`L_dom`|域分类CE，训练`z_dom`表达receiver/day/channel差异|服务域解释和解耦|
|`L_adv`|GRL域对抗CE，使`z_id`难以预测域|降低身份特征域泄漏|
|`L_group_ce`|按receiver/domain group加权的CE或group robustness项|提升弱receiver floor|
|`L_fishr`|约束跨域梯度/风险统计一致性|源域DG稳定性|
|`L_proto`|prototype几何约束|改善Phase2可检索表征|
|`L_zid_compact`|SupCon、角半径、tail CVaR等`z_id`几何约束|收紧同TX簇，扩大类间边界|
|`L_open_world_feature`|类内compact、类间margin、domain align、tail/vacuum指标|源域代理开放边界，不等同真实unknown|
|`L_proxy_unknown`|leave-one-TX-out代理未知边界治理|只在源域模拟未知风险|
|`L_soft_unknown_mixup`|源域不同TX特征混合形成soft virtual unknown|不是目标域unknown监督|
|`L_source_episode`|source-only leave-domain三sigma角壳目标|用于跨域episode压力，不代表Stage2结果|
|`L_pseudo_ce`|对源域无标签样本的伪标签CE|只处理`U_s`|
|`L_entropy`|降低/控制伪标签预测熵|辅助SSL稳定|
|`L_satellite_ce`|源域派生LEO压力视图上的TX CE|是physics-informed stress，不是真实卫星部署证明|
|`L_satellite_consistency`|clean/LEO强视图`z_id`一致性|B02配置中`lambda_sat_cons=0`，主要由satellite CE承担|

### 5.5ADV3B02关键训练参数

|类别|参数|值|
|---|---|---|
|总训练|`epochs`|`200`|
|阶段划分|`label_epochs`|`130`|
|阶段划分|`pseudo_epochs`|`70`|
|open-world warmup|`zid_start`、`ow_start`、`source_start`、`soft_start`、`proxy_start`|`8`、`12`、`20`、`25`、`45`|
|warmup|`warmup`|`25`|
|核心proxy|`proxy_core_q`|`0.90`|
|核心proxy|`proxy_accept_q`|`0.85`|
|核心proxy|`proxy_cvar_alpha`|`0.30`|
|核心proxy|`proxy_core_w`|`0.45`|
|proxy未知|`lambda_proxy_unknown`|`0.0045`|
|proxy未知|`proxy_virtual_count`|`48`|
|proxy未知|`proxy_virtual_mode`|`hard`|
|proxy未知|`proxy_vaccept_w`|`1.00`|
|proxy未知|`proxy_gate_w`|`0.65`|
|proxy未知|`proxy_tail_w`、`proxy_source_w`|`0.20`、`0.20`|
|proxy未知|`proxy_unknown_margin`、`proxy_known_margin`|`0.08`、`0.05`|
|soft未知|`lambda_soft_unknown_mixup`|`0.0045`|
|soft未知|`soft_unknown_mixup_count`|`24`|
|soft未知|`soft_unknown_mixup_order`|`3`|
|soft未知|`soft_unknown_mixup_alpha`|`0.5`|
|soft未知|`soft_unknown_mixup_ce_weight`|`0.60`|
|soft未知|`soft_unknown_mixup_vacuum_weight`|`0.35`|
|source episode|`lambda_source_episode`|`0.0035`|
|source episode|`source_episode_mixup_weight`|`0.75`|
|source episode|`source_episode_radius_cap_deg`|`33`|
|prototype|`lambda_proto`|`0.0032`|
|伪标签|`tau_min`、`tau_max`|`0.92`、`0.97`|
|伪标签|`pseudo_quantile`|`0.86`|
|伪标签|`pseudo_threshold_mode`|`rx_day_quantile`|
|伪标签|`pseudo_domain_gate`、`pseudo_temporal_gate`、`pseudo_strong_agreement`|启用|
|伪标签|`use_ema_teacher`|启用|
|SSL权重|`lambda_u`、`lambda_ent`|`0.16`、`0.01`|
|域对抗|`lambda_adv`|`0.35`|
|Group/FishR|`lambda_group_ce`、`lambda_fishr`|`0.16`、`0.04`|
|satellite stress|`sat_start`|`80`|
|satellite stress|`lambda_sat_cls`、`lambda_sat_cons`|`0.68`、`0`|
|satellite schedule|`sat_schedule`|`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|checkpoint选择|`best_metric`|`joint_safe`|
|安全门控|`enable_joint_safe_guard`、`paic_guard_enabled`|启用|

### 5.6伪标签机制细节

伪标签模块是ADV3B02必须单独作为创新/消融项处理的部分。

流程：

1.在pseudo阶段，对源域无标签样本`x_u`构造weak view和strong view。

2.用EMA teacher优先生成weak view预测；若EMA不可用，则退回当前模型。

3.从`softmax(tx_logits)`得到`conf=max p(y|x_u)`和`pseudo=argmax p(y|x_u)`。

4.按`rx_day_quantile`对不同域分组自适应阈值，阈值被限制在`[tau_min,tau_max]=[0.92,0.97]`，分位数为`0.86`。

5.额外叠加`pseudo_domain_gate`、`pseudo_temporal_gate`和`pseudo_strong_agreement`。只有通过门控的样本才进入`L_pseudo_ce`。

6.对strong view使用伪标签CE，并记录`pseudo_total`、`pseudo_selected`、`pseudo_correct`和`pseudo_conf`。

这一路径的创新点不是“多给模型无标签数据”，而是把源域低标签率RFFI场景中的无TX标签样本纳入receiver/day-aware伪标签治理，尽量避免高置信但域偏置强的伪标签污染`z_id`。

### 5.7ADV3B02训练结果

B02在ADV3机制矩阵中的同row结果如下：

|candidate|overall|strict_udu|receiver_floor|sat_floor|sat_strict_floor|weak_rx|proxy_auc|proxy_vaccept|hard_proxy_accept|bridge_accept|source_overflow|stage2_decision|
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
|`ADV3B02_CORE90_SOFT_E200`|89.1843|84.89|75.55|74.1848|68.7717|`rx7`|0.7494|0.4074|0.2457|1.0|0.4593|是-主诊断候选|

解释边界：

- 该结果说明B02有较强闭集DG、receiver floor和satellite stress鲁棒性。
- `proxy_vaccept`和`hard_proxy_accept`相对前序近饱和proxy接受率显著下降，说明source-only代理未知治理有效。
- `bridge_accept=1.0`和`source_overflow=0.4593`仍是风险，不能写成真实unknown拒识成功。
- 本结果本身不包含Stage2 target-old/target-new适应指标，不能声明`old_acc`、`seen_new_acc`或`H_old_new`由B02单独完成。

## 6.ADV3B02创新模块划分与消融建议

为后续论文和消融，建议把ADV3B02拆成4个模块。这样既覆盖伪标签，也避免把所有技巧混成不可解释的“大模型配置”。

### 模块A：物理先验身份/域解耦主干

组成：

- CV-SincNet物理前端。
- Sinc time path、DAC widely-linear complex path、frequency path、PA memory-polynomial path。
- 双backbone：identity branch输出`z_id`，domain branch输出`z_dom`。
- `domain_head`监督`z_dom`，`adv_domain_head`通过GRL约束`z_id`。
- CosFace/角度margin分类头和`feat_joint`融合。

目标：

- 让`z_id`保留TX硬件指纹。
- 让`z_dom`吸收receiver/day/channel/satellite nuisance。
- 降低跨接收机时身份表征的域泄漏。

建议消融：

|消融项|改法|预期观察|
|---|---|---|
|`no_dac_path`|关闭DAC分支或DAC投影|观察非圆性/IQ失衡建模对receiver floor和hard class的影响|
|`no_pa_path`|关闭PA memory-polynomial分支|观察PA非线性指纹对strict UDU和sat_floor的影响|
|`single_backbone_no_zdom`|去掉domain branch和`z_dom`|检验身份/域解耦是否优于普通CVS分类器|
|`no_grl`|关闭`adv_domain_head`或`lambda_adv=0`|检查`z_id`域泄漏和目标域泛化下降|
|`no_group_fishr`|关闭GroupCE/FishR|检查弱receiver floor是否下降|

### 模块B：源域伪标签一致性SSL

组成：

- EMA teacher或当前模型生成weak view伪标签。
- `rx_day_quantile`自适应阈值。
- `tau_min=0.92`、`tau_max=0.97`、`pseudo_quantile=0.86`。
- `pseudo_domain_gate`、`pseudo_temporal_gate`、`pseudo_strong_agreement`。
- strong view伪标签CE和熵项。

目标：

- 在`rho_label=0.1`下利用源域无标签样本。
- 让无标签样本补充TX身份几何，但避免域偏置伪标签污染。
- 提高低标签RFFI训练下的闭集DG和prototype质量。

建议消融：

|消融项|改法|预期观察|
|---|---|---|
|`no_pseudo`|`lambda_u=0`或禁用pseudo阶段|衡量伪标签对overall/strict/receiver floor的贡献|
|`no_ema_teacher`|只用当前模型生成伪标签|观察teacher稳定性影响|
|`global_threshold`|从`rx_day_quantile`改为global阈值|检查按receiver/day分组是否降低域偏置|
|`no_domain_gate`|关闭domain gate|观察弱receiver伪标签污染风险|
|`no_temporal_gate`|关闭temporal gate|观察时序邻近一致性对伪标签质量的贡献|
|`no_strong_agreement`|关闭strong view一致性门控|检查强增强下伪标签鲁棒性|

### 模块C：core-aware proxy unknown边界治理

组成：

- source-only leave-one-TX-out代理未知。
- B02特有`proxy_core_q=0.90`、`proxy_accept_q=0.85`、`proxy_cvar_alpha=0.30`、`proxy_core_w=0.45`。
- `L_vaccept_CVaR`、component gate、tail quarantine、source safe项。
- hard virtual unknown pool，`proxy_virtual_count=48`。

目标：

- 在没有真实`Y_unknown`参与训练的前提下，用源域代理未知压力减少“类外样本被旧类高置信吸收”的风险。
- 保护known core，避免为降低proxy accept而牺牲旧类闭集DG。
- 产生可推进到真实unknown dry-run的候选，而不是直接声明真实unknown成功。

建议消融：

|消融项|改法|预期观察|
|---|---|---|
|`no_proxy_unknown`|`lambda_proxy_unknown=0`|检查proxy_vaccept/hard_proxy_accept是否回到近饱和|
|`core80_vs_core90`|对比B01/B02或调`proxy_core_q`|观察known core保真和receiver floor变化|
|`accept80_vs_accept85`|调`proxy_accept_q`|观察proxy accept和closed-set精度的权衡|
|`alpha20_vs_alpha30`|调`proxy_cvar_alpha`|观察tail风险惩罚强度|
|`no_component_gate`|关闭component gate权重|检查局部簇边界对hard proxy accept的贡献|

### 模块D：`z_id`几何、soft未知增强和LEO压力视图

组成：

- `zid_compactness_loss`：SupCon、角半径、tail CVaR。
- open-world feature loss：类内compact、类间margin、domain align、tail/vacuum。
- `soft_unknown_mixup`：源域不同TX特征混合形成soft-label virtual unknown。
- source episode三sigma角壳约束。
- 源域派生LEO压力视图和satellite CE。
- prototype导出：`phase2_zid_prototypes.json/.pt`。

目标：

- 让`z_id`成为可检索、可原型化、可少样本注册的身份空间。
- 用soft virtual unknown和source episode压缩边界风险。
- 用LEO压力视图提前暴露星地信道扰动，但不把它写成真实卫星验证。

建议消融：

|消融项|改法|预期观察|
|---|---|---|
|`no_zid_compact`|关闭`lambda_zid`|观察prototype导出质量、strict UDU和qKNN后续性能|
|`no_soft_unknown_mixup`|`lambda_soft_unknown_mixup=0`|检查soft virtual unknown对proxy风险和known core的影响|
|`no_source_episode`|`lambda_source_episode=0`|观察source overflow和跨域角壳稳定性|
|`no_satellite_ce`|`lambda_sat_cls=0`|观察sat_floor和LEO压力鲁棒性|
|`sat_cons_on`|把`lambda_sat_cons`从0提高到小权重|验证clean/LEO一致性是否能补强或引入过约束|

## 7.`qKNNV42`方法细节

### 7.1方法定位

`qKNNV42`不是新backbone，也不是独立`QKNNV42`深度模型类。它是冻结ADV3B02特征上的Phase2轻量部署头：

```text
ADV3B02 backbone(frozen) -> z_id feature export
target receiver K-shot support -> int8 qKNN support memory
query feature -> qKNN/prototype/scenario-aware scoring -> old/new label
```

它服务Stage2-C：

- target-old support用于旧类目标域适应。
- target-new support用于seen-new注册识别。
- query标签只用于离线审计。
- unknown互斥集合不参与本主线，若出现`target_unknown`字段，在N20 HP08L5包中只是历史字段名，实际承载注册新类集合。

### 7.2输入与support memory

输入包括：

|输入|说明|
|---|---|
|主特征NPZ|`features_hardpair_HP08L5_n20.npz`|
|辅助特征NPZ|`features_hardpair_HP08L5_n20_leo_fftlogmag96.npz`|
|old TX|`14-10,14-7,20-15,20-19,6-15,8-20`|
|new TX|20个ManyTx注册新类，如`1-1,1-10,...,8-3`|
|K|当前主结果为`K_old=5,K_new=5`|
|support选择|`support_selection_policy=stable_first`，`seed=421070`|
|query|每类70条，query标签只用于审计|

`QknnMemory`保存：

- int8量化support特征`qfeatures`。
- support标签、old标签、scenario。
- class centroid。
- class radius/Mahalanobis/EVT/oldness阈值。
- source-old prototype shrinkage元数据。
- margin/score阈值。
- 存储计数和压缩状态。

量化公式：

```text
z_i = normalize(feature_i)
q_i = clip(round(127 * z_i), -127, 127).astype(int8)
support_i ≈ normalize(q_i / 127)
```

该设计避免保存原始support IQ，也避免重训backbone。主结果JSON记录`stored_raw_support_count=0`、`stored_quantized_support_code_count=130`、`stored_class_prototype_count=26`。

### 7.3打分公式

对query特征`x`：

```text
z = normalize(frozen_backbone(x))
s_i = z^T normalize(q_i / 127)
```

对每个候选类`c`，基础分数可写为：

```text
local_c = mean_topm({s_i | y_i=c})
proto_c = cosine(z, centroid_c)
score_c = (1 - λ_proto) * local_c + λ_proto * proto_c + b_old * I[c in Y_old]
```

其中：

- `topm`控制每类局部近邻均值。
- `λ_proto=proto_mix`控制prototype混合。
- `b_old=old_bias`给旧类小幅先验，防止新类注册时旧类遗忘。
- scenario-aware mask在support场景足够时优先同LEO场景support。
- auxiliary view先通过support LOO可靠性门控，再决定是否混入主分数。

V42门控思想：

```text
primary_loo = support LOO accuracy in ADV3B02 z_id space
aux_loo = support LOO accuracy in auxiliary LEO-sketch/fft-logmag space
gate = min(mean_gate, floor_gate, absolute_floor_gate)
effective_aux_score_weight = aux_score_weight * gate
score = (1 - effective_aux_score_weight) * primary_score
      + effective_aux_score_weight * aux_score
```

当辅助视图弱于主视图时，`effective_aux_score_weight`会自动降低。历史报告中多处显示旧LEO-sketch被V42门控拒绝，这是正确的负向保护，不应写成辅助视图必然增益。

### 7.4阶段性成果行

当前最可写入阶段成果的是N20 HP08L5注册新类包中的K5 high-floor候选。该行来自：

```text
E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v53_fftlogmag_20260706\local_v55_diagnostics_20260706\k5_strict_seed421070_floor_param_best_predictions_20260707.json
```

同row结果：

|seed|K_old|K_new|old_acc|min_old|seen_new_acc|min_new|H_old_new|old类数|new类数|support code|query sha|verdict|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|421070|5|5|94.52%|85.71%|90.14%|81.43%|92.28%|6|20|`a84b66e28e565c52`|`75c99f6361810ca9`|Phase2 K5旧类适应+seen-new注册候选|

该行关键参数：

|参数|值|
|---|---|
|`transform_mode`|`diag_whiten_fisher`|
|`transform_strength`|`0.1`|
|`topm`|`1`|
|`proto_mix`|`0.45`|
|`aux_score_weight`|`0.34`|
|`effective_aux_score_weight`|`0.34`|
|`old_bias`|`0.001`|
|`mutual_only`|`true`|
|`scenario_aware`|`true`|
|`balanced_assignment`|`true`|
|`role_balanced_assignment`|`true`|
|`labelprop_weight`|`0.025`|
|`labelprop_k`|`10`|
|`labelprop_alpha`|`0.76`|
|`labelprop_temperature`|`0.05`|
|`labelprop_rounds`|`8`|
|`scenario_residual_weight`|`0.5`|
|`scenario_residual_scope`|`new`|
|`stored_quantized_support_code_count`|`130`|
|`stored_raw_support_count`|`0`|
|`stored_class_prototype_count`|`26`|

逐类结果：

|role|TX|acc|
|---|---|---:|
|old|`14-10`|94.29%|
|old|`14-7`|85.71%|
|old|`20-15`|100.00%|
|old|`20-19`|87.14%|
|old|`6-15`|100.00%|
|old|`8-20`|100.00%|
|new|`1-1`|85.71%|
|new|`1-10`|88.57%|
|new|`1-11`|94.29%|
|new|`1-12`|81.43%|
|new|`1-14`|81.43%|
|new|`1-15`|84.29%|
|new|`1-16`|94.29%|
|new|`1-18`|95.71%|
|new|`1-19`|91.43%|
|new|`1-2`|87.14%|
|new|`10-10`|92.86%|
|new|`11-10`|92.86%|
|new|`18-5`|95.71%|
|new|`19-3`|84.29%|
|new|`2-13`|84.29%|
|new|`2-5`|92.86%|
|new|`3-8`|97.14%|
|new|`4-10`|94.29%|
|new|`8-18`|95.71%|
|new|`8-3`|88.57%|

该行是“同row、同split、同K”的主证据。不能把它与其他候选的unknown FAR或单项最大值拼接。

### 7.5V42基线与后续参数面的关系

报告中应区分初始V42基线和后续V42线路参数面：

|设置|policy|topm|old_acc|min_old|seen_new_acc|min_new|结论|
|---|---|---:|---:|---:|---:|---:|---|
|K5,N14|V42|4|92.00%|80.00%|90.10%|73.33%|接近但未过新类floor|
|K5,N20|V42|4|92.00%|80.00%|79.80%|69.33%|20新类floor不足|
|K10,N14|V42|4|91.90%|82.86%|90.20%|68.57%|新类floor不足|
|K10,N20|V42|4|91.90%|82.86%|84.64%|72.86%|新类floor不足|
|K5,N20 high-floor行|V42线路参数面|1|94.52%|85.71%|90.14%|81.43%|阶段性候选，但仍需oracle-free support selection|

因此写法应为：`qKNNV42`给出了冻结特征压缩support-memory头的基本机制；当前最好行属于该机制路线上的参数面/支持集敏感性证据，证明该表征空间中存在同时保护旧类和20个seen-new新类的K5注册候选，但下一步必须把support选择和质量门控做成不依赖query真值的注册期机制。

### 7.6qKNNV42相对现有RFFI的创新点

相对常见RFFI闭集分类、普通微调和传统KNN，qKNNV42的创新主要在协议组合和部署状态设计，而不是发明全新的距离函数。

|维度|现有RFFI常见做法|qKNNV42的具体差异|
|---|---|---|
|任务协议|固定receiver闭集分类，或源/目标域离线DA评估|同一`R_t`LEO目标域内同时做target-old保留和target-new注册|
|模型更新|常见是重训/微调分类头或全模型|冻结ADV3B02 backbone，只更新support memory和轻量score状态|
|部署状态|可能保存样本、logits、full checkpoint或分类器参数|保存int8量化support code、class prototype、半径/阈值/少量标量，不保存原始support IQ|
|新类学习|常见闭集RFFI不处理新TX注册|把target-new support直接注册进class memory，支持seen-new识别|
|旧类保护|新类注册可能导致旧类遗忘|通过old support、source-old prototype shrinkage和old bias保护`Y_old`|
|场景扰动|常见把信道/receiver差异作为噪声或域标签|把LEO scenario作为support选择和残差补全条件|
|质量估计|常见离线调参依赖query表现|V42引入support-only LOO门控评估辅助视图是否可用|
|工程部署|常见模型大、状态重、不可解释|qKNNV42状态小、可审计、可按support指纹复现|

更准确的论文表述应是：

```text
qKNNV42是面向CVS Stage2-C的压缩support-memory少样本注册头。它把冻结物理先验CV-SincNet学到的身份表征转化为目标接收机域内的量化近邻记忆，通过类内topm近邻、prototype混合、旧类锚定、scenario-aware仲裁和support-only辅助视图门控，在不重训backbone的条件下同时评估target-old保留和target-new seen-new注册。
```

### 7.7其他深度学习/机器学习中的类似思想

qKNNV42与多类已有思想相似，但在CVS协议中的组合方式和边界不同：

|类似方法|相似点|qKNNV42差异|
|---|---|---|
|KNN/nearest neighbor|基于embedding距离做最近邻分类|使用冻结RFFI身份表征、int8压缩support、旧/新类角色、LEO scenario和support指纹|
|Nearest Class Mean|使用类中心分类|qKNNV42同时混合类内topm近邻和prototype，不只用均值|
|Prototypical Networks|few-shot support形成类原型|qKNNV42不训练episodic backbone，backbone已由ADV3B02冻结；重点是部署期support memory|
|Matching Networks|query-support attention|qKNNV42的topm和label propagation有检索/传播味道，但不做端到端attention训练|
|iCaRL/增量学习prototype|新类注册、旧类prototype保留|qKNNV42不存原始样本，不做分类器增量训练，强调目标接收机K-shot|
|Mahalanobis/open-set score|类协方差/半径用于风险估计|在V42路线中是辅助风险信号，不构成Phase3成功|
|OpenMax/EVT|用尾部分布做open-set拒识|qKNNV42代码可存EVT/半径阈值，但本报告不把unknown拒识写成已成功|
|Conformal/support LOO|用校准集或LOO估计可靠性|qKNNV42的support LOO门控只用support，不用query真值；当前support quality门控也有负证据，不能夸大|
|量化检索/边缘ANN|用int8/压缩embedding降低存储|qKNNV42将其用于RFFI在轨目标域few-shot support memory|

## 8.Phase3 unknown诊断负证据

必须把unknown/open-set结果单独列为负证据。当前不能把它们写成成功。

完整互斥unknown诊断报告：

```text
E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_v42_stage2c_unknown_20260707\report.md
```

关键结论：

|诊断项|结果|解释|
|---|---|---|
|known-vs-target_unknown AUROC|约0.6006-0.6045|已知/未知几何重叠强|
|FPR95|约0.945-0.9475|高召回已知时误接未知严重|
|PCET最高old行|`old_acc≈0.4023`、`min_old≈0.1167`、`seen_new_acc≈0.1117`、`unknown_reject≈0.6373`、`FAR≈0.2409`|known和unknown无法同时满足|
|最高unknown_reject行|`unknown_reject≈0.9845`、known几乎被拒|不能作为部署结果|
|直接qKNN小扫|known平均最高仍低，seen-new多为0|冻结特征+后处理不能补齐Phase3|

结论：qKNNV42在Phase3互斥unknown协议下是`NON_DEPLOYMENT_DIAGNOSTIC`。它暴露出当前表征空间中known/unknown重叠过强，需要表示重建或新的Phase3机制；不能用低FAR单项结果掩盖old/new不足。

## 9.推荐论文/报告写法

建议主文使用如下结构：

1.问题定义：地面弱标注跨接收机DG到目标接收机LEO少样本注册。

2.模型：CV-SincNet物理先验+`z_id/z_dom`解耦。

3.Phase1训练：ADV3B02的source-only低标签训练机制，包括伪标签SSL、proxy/soft未知边界、source episode和LEO压力视图。

4.Phase2部署头：qKNNV42压缩support-memory，冻结backbone，支持target-old和target-new。

5.结果：先报告B02 Phase1指标，再报告qKNNV42同row old/new结果，最后单独报告unknown负证据。

6.消融：按4个ADV3B02模块和qKNNV42组件逐项拆分。

推荐摘要句：

```text
我们将CVS分解为source-only物理先验表征学习和目标域轻量注册两个阶段。`ADV3B02_CORE90_SOFT_E200`在ManySig源域低标签协议下学习`z_id/z_dom`解耦表征，并通过源域伪标签、core-aware proxy unknown和LEO压力视图提高跨接收机鲁棒性。`qKNNV42`在冻结`z_id`空间中构建int8压缩support memory，把target-old保留和target-new seen-new注册统一为目标接收机域内的少样本检索问题。在N20 HP08L5注册新类包上，当前同row候选达到`old_acc=94.52%`、`min_old=85.71%`、`seen_new_acc=90.14%`、`min_new=81.43%`。真实unknown拒识仍是Phase3诊断负项，不作为本阶段成功声明。
```

## 10.复现与证据索引

|证据|路径|用途|
|---|---|---|
|项目协议|`E:\type10-7\项目.md`|CVS科学场景、数据协议、Stage2/Phase3边界|
|AGENTS规则|`E:\type10-7\AGENTS.md`|Git、N607、报告、中文排版和安全规则|
|ADV3B02主报告|`E:\type10-7\automation_reports\CV-SincNet\phase1_adv3_mechanism32_queue_20260701\report.md`|B02身份、数据、候选和prototype导出|
|ADV3B02分析|`E:\type10-7\automation_reports\CV-SincNet\phase1_adv3_mechanism32_queue_20260701\full_analysis_20260702.md`|Phase1边界、B02结论和不可声明项|
|ADV3B02候选表|`E:\type10-7\automation_reports\CV-SincNet\phase1_adv3_mechanism32_queue_20260701\adv3_m32_candidate_summary.csv`|同row指标|
|ADV3B02训练快照|`E:\type10-7\code\snapshots\phase1_adv3_mechanism32_queue_20260701\launch_phase1_adv3_mechanism32_queue_20260701.sh`|训练参数与候选variant|
|CVS模型|`E:\type10-7\code\model.py`、`E:\type10-7\code\model_dual_cvsincnet.py`|CV-SincNet和双分支解耦结构|
|SSDG训练|`E:\type10-7\code\SSDG\train_ssdg.py`|伪标签、loss、guard和日志字段|
|qKNN主实现|`E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py`|`QknnMemory`、int8量化、KNN打分|
|qKNNV42策略实现|`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_support_metric_qknn_probe.py`|V42策略、support LOO门控、topm/prototype/labelprop/scenario残差|
|qKNNV42主报告|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\report.md`|V42矩阵、high-floor行解释、字段边界|
|qKNNV42最佳JSON|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v53_fftlogmag_20260706\local_v55_diagnostics_20260706\k5_strict_seed421070_floor_param_best_predictions_20260707.json`|当前同row指标和support/query指纹|
|逐query审计|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v53_fftlogmag_20260706\local_v55_diagnostics_20260706\k5_strict_seed421070_floor_param_best_predictions_rows_20260707.csv`|逐样本预测证据|
|unknown负证据|`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_v42_stage2c_unknown_20260707\report.md`|Phase3/unknown互斥诊断负结果|

## 11.下一步建议

1.把ADV3B02的4模块消融做成正式矩阵：`backbone/disentangle`、`pseudo SSL`、`core proxy unknown`、`z_id geometry+soft unknown+LEO stress`。

2.把qKNNV42拆成可部署组件消融：`int8 qKNN`、`prototype mix`、`old bias`、`scenario-aware`、`support LOO aux gate`、`labelprop`、`scenario residual`。

3.优先解决qKNNV42的oracle-free support selection。当前强行来自`seed=421070`，证明空间内存在强support组合，但还不是注册期自动选择机制。

4.不要继续把unknown FAR作为Phase2主线目标。真实unknown应另开Phase3表示学习或拒识机制，不能用qKNNV42后处理硬补。

5.后续所有报告保持同row表格：candidate、K、receiver、old_acc、min_old、seen_new_acc、min_new、unknown诊断、support/query指纹和verdict必须同行出现。
