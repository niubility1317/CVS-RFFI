# 动力博弈历史实验专题：CORE90、DAOT/FastTrust与XUC融合

本章核对日期为2026-09-27，主体证据窗口截至2026-09-14。它解释历史配置、实际执行与已保存评分，不把旧RUNNING当作当前状态。9月17日后response矩阵、pure_game及9月27日远端状态由总报告另章处理。本次没有训练、推理、重新连接truth评分或远端修改。

## 1.可成立的结论及其适用范围

历史证据支持完整Extragradient（EG）在特定CORE90对照中改善LEO泛化；不支持“所有动力博弈方法都有效”。V2完整EG相对普通更新在adv=0和adv=.35两种背景的三个seed均提升；只预测对抗头的lookahead在adv=.35下三个seed均下降。它们是不同算法，不能用一个“动力博弈”标签合并。

9月14日前融合矩阵没有完成同一承载、同一数据/随机流、同一预算的“原生有效DAOT+RC4 × 每步完整EG”四格。因此，XUC融合低于A1不能证明两者天然不相容。FULL的C2只在约2.4%的主步修正，C*没有实际修正。DR7还带有已确认的预算和U分支梯度语义漂移。

可靠性探针的问题独立于优化器收益。域头恢复失败后把负gap截成0，不是“已经消除了域信息”；可靠C*拒绝失败恢复是正确的证据处理，但零动作不能作为有效控制策略优劣的实验证据。即使完整EG提高准确率，也不能由此断言收益全部来自消除域博弈旋转，因为关闭ADV后仍观察到收益。

主证据包为[9月14日综合报告](../../daot_fasttrust_game_20260914/report.md)，以下以其目录为`B`。配置全集见[B/configuration_appendix.md](../../daot_fasttrust_game_20260914/configuration_appendix.md)。引用的完整路径均保留在[B/source_manifest.csv](../../daot_fasttrust_game_20260914/source_manifest.csv)，本章不靠对话回忆补写实验事实。

## 2.数据与计数：同名模型不等于同一零点

共同物理契约为ManySig equalized IQ、长度256、6个注册TX（ID0至5），源RX为1/3/4/6/8，源day索引1/2/3，对应03_08、03_15、03_23；源L/U/V为6300/56700/27000。U是源接收机的隐藏TX标签数据，不是目标无标签适配池。V只读校准。目标RX为0/2/5/7/9/10/11，day为0/1/2/3，每场景168000条；每模型4场景672000次决策来自168000条物理IQ的四种视图，并非672000个独立样本。

LEO均值为三个LEO场景accuracy等权平均。跨seed标准差使用样本标准差；XUC/FULL只有seed392005，不应报告其训练稳定性显著结论。模型seed、split seed、data order、augmentation/evaluation随机流必须分别比较：GAME V2模型seed392005/6/7而split/eval为392002，XUC相应流为392005。不能把它们终点差值解释为单一机制效应。

|历史参照|clean准确率%|LEO均值%|主要不可互换因素|
|---|---:|---:|---|
|FCR ADV3B02|76.2268|60.1397|早期实现/AMP/几何与选模差异|
|GAME V1 B0|74.3643|61.8022|FP32、普通49步/轮、U跨epoch窗口|
|CROSS U0，统一168000口径|74.7321|59.0758|旧198000总体混入30000条已见RX/未见日期|
|XUC M00|78.4619|62.5048|XUC承载及随机流|

U0旧198000总体的clean为77.3005%，不能把比74.7321%高出的部分当模型改进。原历史`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`的best在E194，不是当前scratch的固定final E200。其旧target接触及joint_safe选模不合当前checkpoint契约。A1早期两条继承结果必须保留但标为不能支持干净泛化；FULL记录则是scratch_only、无外部checkpoint、E20同run EMA生成anchor。参考依据为[B/sources/BASELINE/expanded_report.md](../../daot_fasttrust_game_20260914/sources/BASELINE/expanded_report.md)及项目协议4.4节。

## 3.CORE90中的博弈场与求解器分工

CORE90是模型、损失、课程、伪标签和优化设置的组合，不是一个单独优化器。身份分支`z_id`与域分支`z_dom`均为160维，身份头预测6TX，对抗域头预测5RX×3day的15域。GRL使对抗头最小化域CE，身份编码器接收相反方向；域分支自身域CE仍正常最小化。基础场还含TX CE、orth、cons、group CE、Fishr logit梯度代理、prototype、open-world geometry、compact、proxy unknown、mixup、episode及卫星CE。

完整EG在原点算完整耦合场，临时移动参数，再在同批数据和冻结教师/路由语境下重算完整场；恢复原参数与优化器状态，用第二场梯度接受一次AdamW更新。每步两次场评估不是两次正式optimizer更新。工程实现包含AdamW矩、梯度裁剪、EMA及离散路由，不能直接套用简单双线性SGD收敛结论。Heun取两场平均，Optimistic使用历史梯度，head-lookahead仅临时移动对抗头，C2/C*决定动作或课程，均不等价于每步完整EG。

## 4.GAME V1、V2与高学习率实验

### 4.1.V1：25条已评分行与真实失败

V1矩阵26行中25行有目标评分；B6 Heun缺完整目标评分，应为空而非0。另有C4因缺合法donor而未创建，不是完成实验。B4在head交替2次后主步排除head，目标全部塌缩为TX1，四场景均16.6667%；B4_fixedk改变为额外head步后仍作耦合更新，其63.8927%不是同一算法简单重跑。

|V1行|LEO均值%|实际对象|
|---|---:|---|
|B0|61.8022|普通simultaneous AdamW|
|B2|63.7827|lambda_adv=0|
|B3_lr_high|64.5829|普通更新lr=.0004|
|B5|65.4454|完整EG|
|B8|65.5558|仅对抗头lookahead|
|B7|57.9784|Optimistic|
|S3|65.1006|correction-only控制|
|S4|63.7690|两类控制动作|
|C2|62.3562|控制加能力课程|

旧C2的恢复学习率.02、40步，31次monitor CE全比在线头差，却把负gap裁为0；其hard LEO课程推迟至E189附近，有效困难曝光42201，对照S4为409039，仅10.32%。这证明课程是明确混杂，不能把C2−S4归给求解器。H1隐式响应40次尝试仅13接受，另12次monitor拒绝、15次CG未达预算；J1局部Jacobian20次13有效，均不能包装为全网精确响应或全局稳定性证明。

### 4.2.V2：按seed配对才看得清

V2设计新增fit/monitor capture-group分离、恢复有效性、独立动作时钟和负对照。但正式21行主要是固定课程、无动态审计的求解器对照：6方法×3seed加3条单seedLR参考。可靠控制器存在于代码，不代表它在这21行实际工作。

|方法|clean均值±SD%|LEO均值±SD%|解释|
|---|---:|---:|---|
|A，普通adv0|75.907±0.868|60.146±1.632|无ADV基线|
|B，普通adv.35|74.684±1.748|61.166±2.396|ADV平均+1.0208pp，seed方向不全一致|
|C，head-lookahead adv0|与A相同|与A相同|应退化一致的负对照通过|
|D，head-lookahead adv.35|75.391±0.633|58.418±1.731|相对B三seed均下降|
|E，完整EG adv0|74.700±4.228|63.378±2.291|相对A三seed均提升|
|F，完整EG adv.35|76.851±1.204|63.745±2.182|相对B三seed均提升|

以seed392005/392006/392007顺序，D−B为−4.5317/−1.1252/−2.5883pp，均值−2.7484pp；E−A为+4.5046/+2.9933/+2.1994pp，均值+3.2324pp；F−B为+0.1456/+5.1194/+2.4722pp，均值+2.5791pp。F−E只有−0.7125/+1.0571/+0.7577pp，平均+0.3675pp。完整EG总体收益明确比“EG使ADV必然有效”的论断更有证据。

V2源开发的恢复fit CE由2.721549降至.128511，monitor却由2.729933升至4.533828，不能作为可靠CORRECT依据。该问题与完整EG固定求解器测试应分开描述。详见[B/sources/GAME_V2](../../daot_fasttrust_game_20260914/sources/GAME_V2/target_report.md)。

### 4.3.高LR六行已完成，不能沿用旧发布状态

|完整EG lr=.0004|clean均值±SD%|LEO均值±SD%|
|---|---:|---:|
|adv0，3seed|75.499±0.424|65.222±1.749|
|adv.35，3seed|77.091±0.997|66.555±1.147|

高LR两背景的配对ADV增量为+2.6635/+1.0492/+0.2863pp，平均+1.3330pp。普通高LR强源参考只跑单seed，clean79.7256%、LEO67.5992%，不是完整EG三seed结果，也不能据此认定普通高LR稳定胜过EG。六行均未同时启用DAOT+FastTrust；与F-A1的68.7%跨契约相减不能估算机制排名。原证据为[B/sources/EG_HIGH](../../daot_fasttrust_game_20260914/sources/EG_HIGH/target_report.md)。

## 5.A1旁支与“原生有效DR”的限定

原生A1有效DAOT是two-view mean、feature/logit蒸馏；prototype参数可残留.2，但传入矩阵None，不产生该项梯度。RC4主要使用H/P，N、anchor和U卫星hard关闭，U forward GRL=0，每轮222步、E200共44400步。FULL全扩展改成three-view robust，并加prototype/relation/tangent/nuisance/fingerprint、N/anchor/U卫星hard；这不是同一方法补齐记录，而是新增算法组合。

|A1历史对照|LEO均值%|能说明什么|
|---|---:|---|
|B0_FIXED|67.8315|原生机制参考|
|D0_NO_ORBIT|63.7173|保留RC4去orbit；DAOT条件增量+4.1143pp|
|D1_THREE_VIEW|68.3034|增教师视图|
|D2_PHYSICAL_ORBIT|66.7534|robust扩展相对D1下降1.5500pp|
|D3_TANGENT|67.9089|相对D2回升，未证明所有扩展都有益|
|E0_RESPONSE_ONLY|68.7149|名字含response不等于完整预测式ECRS|
|X2_E400|70.0337|预算不同，E300峰值70.3621不能代替final E400|
|R3_CLEAN_RX_E400|69.6359|长预算探索|

历史A1全覆盖60行含失败/替代，33条已评分；两条继承A1有target-contact边界。原生R3 scratch参考LEO68.5133%、fast sequential68.2323%，属于干净初始化参考。F0/F1在不同家族重复命名，必须同时写run与row。已有FastTrust×DAOT独立全因子矩阵仍不足，因此D0消融只证明DAOT在RC4背景下的条件作用。

9月14日另有[原生DR+完整EG本地实施验收](E:/type10-7/automation_reports/CV-SincNet/response_games_prepare_20260914/previous_native_dr_eg_report.md)：18个R0候选、9个R1模板，明确PREPARED_NOT_LAUNCHED、launch=false，只进行了本地合成验收，不能补成正式实验。后续response/pure_game归另章。9月18日后practical4、residual-ratios属于DAOT/RC4工程延伸，除非配置实际启用博弈机制，不纳入完整EG成果数。

## 6.XUC三批融合：INITIAL15、DR7、FULL9

X是clean跨RX同TX拉近/异TX间隔约束，lambda=.05；Ux是单位记录特征的TX×RX双中心交互残差，lambda=.01。Ux中的U不是源无标签池U。grid为P4×Q4×K2、每batch4块共128；可消除交互不意味着消除RX主效应，也不保证保留身份。BN冻结上下文在本模型检测到0个BatchNorm，不能当实质收益来源。

INITIAL15中M00普通CORE90，M01仅grid，M02/M03/M04分别X/Ux/C2，M05为X+Ux，M06/M07/M08为加C2的组合，M09原生A1，M10=A1+X，M11普通+C2，M12/M13为C*加不同票据课程，M14为被动C*。M09/M10为44400步，其他13行9800步。

M09=68.3260%、M10=67.6865%，X增量−0.6395pp，和另一A1家族X1−X0的+0.3921pp不同，说明上下文依赖。grid内部X与Ux交互M05−M02−M03+M01=−2.0690pp；C2包与X+Ux包交互M08−M05−M04+M01=+1.7252pp。联合低于基线不等价于负统计交互，且单seed不能证明稳定交互。

|机制/预算|原生有效A1|DR7移植|FULL修订|
|---|---:|---:|---:|
|每轮主步|222|49|222|
|E200主步|44400|9800|44400|
|U累计曝光|11340000|2502992|11340000|
|DAOT执行主步|39960|8820|39960|
|U身份GRL|0|1（偏离）|0|
|扩展目标|原生有效集合|原生有效集合|多数行全扩展|

DR7覆盖了全部56700个U物理ID；问题是仅约44.14次U池遍历，不能误写“只使用22%的U样本”。相同域CE=.482625的验证中，GRL偏移使z梯度从0变.03190而域头梯度仍约.730178。七行LEO为52.0633%至55.2825%，是真实失败，但不能认定原生A1保真融合失败。总墙时较短来自步数变少；每步耗时反而约1.34至1.42秒，对照原生约1.05至1.08秒。

FULL训练发布commit为`e7dbe643f5b85062e8f4e2de6d711a5447c7a422`，修复预算与GRL同时加入全扩展DR。F-A1为原生有效损失承载参考；F-M11为full DR+C2；F-M05为full DR+X+Ux；F-M08再加C2；F-M14/M12分别被动C*/主动C*加能力票据；F-M13主动C*固定票据；F-M07为full DR+Ux+C2；F-M00匹配222步/LR的普通CORE90。缺少full DR alone、无X/Ux/C的行，因此F-M11−F-A1同时改变扩展目标和C2。

## 7.FULL截至9月14日的终点与内部证据

|行|clean%|LEO均值%|相对F-A1 pp|最弱TX LEO%|
|---|---:|---:|---:|---:|
|F-A1|78.2488|68.7000|0|35.5881|
|F-M11|77.6762|67.7877|−0.9123|34.5655|
|F-M08|78.2637|67.1022|−1.5978|29.0988|
|F-M05|77.2506|66.0157|−2.6843|28.6179|
|F-M14|76.2905|65.7385|−2.9615|29.7798|
|F-M12|77.2619|65.2488|−3.4512|31.4345|

F-M08比同full DR+X+Ux背景的F-M05高1.0865pp，反驳“C2总恶化”。F-M08的clean比F-A1高.0149pp而LEO更差，反驳“clean相近即鲁棒性相近”。融合多行改善TX3/部分TX5而损害TX0/1/4；这与身份信息被过度约束相容，但没有特征干预证据证明哪个硬件指纹被删除。

每行三LEO累计504000次决策，相对F-A1，F-M11 rescue/harm=14736/19334，F-M08=25946/33999，F-M05=24216/37745，F-M14=21788/36714，F-M12=21355/38749。联合确有救回的样本，只是损坏更多。窗口和多场景不是独立训练重复，不能用几十万决策制造伪统计确定性。

|行|H/P/N累计选择|C2修正主步|恢复可靠性|
|---|---|---:|---|
|F-A1|380407 / 923119 / 0|0|无控制|
|F-M11|106588 / 2560971 / 907491|1056（2.3784%）|148次legacy恢复全更差|
|F-M08|114208 / 2973298 / 1042017|1076（2.4234%）|149次legacy恢复全更差|
|F-M05|104282 / 0 / 3674712|0|178次C*全部无效|
|F-M14|81589 / 3894801 / 1552506|0|178次C*全部无效|
|F-M12|102293 / 3623542 / 745698|0|178次C*全部无效|

累计选择是重复路由事件，不是唯一样本，不同分支也不可直接求和当监督样本总量。F-M05的P为0是实际失活，不能因为配置开启就写partial已工作。C*无效不意味着每次monitor点估计都更差：其判据含bootstrap上界与可靠性门槛；核验C*全部无效应看`valid=False`，不能以恢复CE单一大小替代。CSV的legacy `valid=True`仅表示legacy记录被纳入，不等价于恢复可靠性通过。

FULL的困难卫星曝光不能沿用V1解释：E80起，固定票据行有效卫星selected为2686483，C2的F-M11/F-M08为2749015，后者高2.33%。票据次序、origin_epoch驱动的LR/损失阶段与live_epoch驱动的校准/EMA仍可不同，总曝光匹配不等于路径一致。

全扩展45个定期梯度探针中orbit z/logit/proto/relation及nuisance约40次非零，tangent约31次，fingerprint抽查为0。后者只能写“抽查未观测有效梯度”，不可推成全程零或已保护指纹。F-M11 step44000：DAOT总范数20.5262，RC4总4.5053；orbit prototype13.5716、logit8.0942，高于系数直觉；全局clip=5会耦合有效步长。梯度范数不能相加为能量分解，不同参数范围不能直接比。X/Ux余弦均值约+.23至+.24，不支持“二者始终冲突”；未有同参数范围、同主步的DAOT与ADV/TX全量余弦，不能将梯度对冲写成已证实原因。

## 8.因果解释分级

已证实：DR7预算/GRL移植漂移；FULL修复与多分支扩展捆绑；C2恢复失败却截gap；C*没有动作；RC4路由显著分叉；新增prototype/logit产生较强梯度；XUC未构成原生DR×完整EG四格。

有运行迹象但未分离贡献：CORE90几何、X/Ux、DAOT与GRL共同抑制变化，可能连身份相关交互一起压平；教师—学生—路由闭环可能放大错误或改变监督覆盖；标量loss EMA归一化不是梯度均衡，可能经clip/AdamW矩改变其他项；尾段骨干LR降至1e−6而头仍2e−5，改变双方时标。无法从现有终点给各项分配准确率损失。

不支持：source V接近98.5%即训练没问题；domain CE接近log(15)即不可读域信息；所有配置启用即所有机制有效；C*主动行差于被动行即C*修正有害；共享GPU导致精度下降；target历史峰值/最佳行仍是新盲确认。四格交互`(S_EG+DR−S_DR)−(S_EG−S_base)`只能在匹配契约内计算，本章不跨家族拼出一个协同系数。

## 9.本次独立复核与历史状态边界

本次完整解析113条评分记录，家族A1/GAME_V1/GAME_V2/EG_HIGH/XUC为33/25/21/6/28；逐行重算LEO均值的最大误差为1.43×10⁻¹⁴pp。113是报告行记录数，不是独立训练数。同一模型在多个对照报告中可复用。

完整解析配置附录551262字符及20个原始JSON配置块、10028条config键值记录、1722条epoch-mechanics记录、480条gradient-probe记录、1128条audit记录、123条activation记录、31条XUC覆盖记录，并检查两份完整gzip采集证据。`full_live_audit.json.gz`六个完成行各200个epoch、接受主步和44400、mean_loss无非有限数、全部source V日志标记只读。三个partial行的完整epoch日志为178/177/163轮；action快照步数39616/39456/36365包含当前未闭合轮次，不应强行与完整epoch步数39516/39294/36186相等。另一次mechanics采集稍晚，包含179/178/165个epoch桶，不能把两时点字段拼成一张伪同时快照。

9月14日历史包对F-M00/F-M07/F-M13没有final目标分数。对实验索引及本地automation_reports中相关报告/summary/status文件作针对性检索，未找到后续终评补齐；此结论仅说明本地历史证据覆盖，不能推出9月27日它们仍运行或失败。总报告新增远端快照应独立标时间并覆盖当前状态，不改写9月14日快照。

## 10.发布资料范围与最小补齐项

旧B目录作为历史证据包可发布：source_manifest列387条路径全部存在；386条非空SHA256均独立匹配。唯一空SHA项为`evidence/XUC_DR7_full_run_audit.json.gz`，已注明gzip转换；本次解压29046112字节与原JSON完全一致，压缩后SHA256为`350d3993edf32718b60be45f3b0e361ff5ae1e5caee4bffec043f91b04b806f5`。该项不是损坏。manifest组成271个code_snapshot、88个sources、20个configs、8个evidence，总记录字节28335931。新总发布清单应记录转换与现存文件，而非盲目把空值当校验失败。

最小保留集合为B内主报告、配置附录、source_manifest、validation.json、tables全目录、sources全目录、configs全目录、evidence全目录、figures及code_snapshot。还应补齐包外三份小型完整采集聚合证据：

- `automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914/full_live_audit.json.gz`（1275181字节）。
- `automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914/full_mechanics.json.gz`（2011692字节）。
- `automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914/a1_all33_recount.json.gz`（354556字节）。

旧code_snapshot只有`cvsrffi/`与`SSDG/`两目录，适于机制审计，不能单独声称是可直接启动的完整实验包。应由总发布包补充相应历史Git版本的训练入口、launcher、依赖说明及报告生成/评分工具，或清楚链接已发布的对应commit。不要把当前改过的文件伪装为9月14日执行源码。建议保留原FULL发布报告、独立六行评估报告和V2/EG_HIGH训练与目标评估报告，建立“设计→配置→执行commit→epoch/动作→固定预测评分→分析”的导航。

大体积IQ、checkpoint、逐样本预测及原始逐步日志不因本次整理重新复制进Git；使用原始路径、已有远端位置与体积说明。会话导出用于定位上下文，不是方法科学证据；无需将完整个人对话文本上传。9月27日新采集证据应单独归档，使读者可区分历史partial快照与后续闭合状态。
