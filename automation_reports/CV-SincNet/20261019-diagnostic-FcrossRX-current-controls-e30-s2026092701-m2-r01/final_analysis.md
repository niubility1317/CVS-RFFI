# P79 结果：当前基座的剩余损失不能归结为新增噪声

状态：ANALYZED_SOURCE_ONLY。实际代码 308f4ca53bf87cb23c47cbcd43ec5b1da2ae8b56，起点 f2f19130831ed2c6db220062deec913c6df66798。两个固定P43 FINAL30 student/defaultcos30，种子2026092701，源V1800×2既有信道draw×3场景×11条件已完成；没有训练、新分类头、新信道或正式测试访问。两draw不是两模型seed。

## 核心判断

新增噪声是本轮最大单项可释放因素，但并不能解释全部剩余损失。AS30的源low urban从59.083333%升到固定接收状态去新增噪声后的68.138889%，重算状态后70.166667%；同源clean为98.333333%。因此“只要进一步去噪就能恢复clean”的假设未获支持。去掉噪声也引入213个新错误，净修复326个，不是所有包都单调改善，更不是理论上限。原录音中的未知噪声仍存在。

flat_H/fixed在mid/low分别提高3.361111／1.638889个百分点；carrier/fixed只有0.083333／0.305556，注入RX IQ残差移除只有0／0.027778。没有证据把CFO、RX IQ补偿或全分支门控一致性当下一主线。控制保持其他损伤不变，不是噪声去除后的再增益，不能相加作可达性能。

reflect→edge的人工边界干预反而使AS mid/low下降10.888889／8.000000个百分点，且大部分损失来自仍有较好新增噪声比的组，提示边界改变会破坏原本可用的证据，不表示edge能修复模型。固定边界敏感也不能推出边界混合训练有益：已有P26完整native practical reflect/edge 1:1训练的纯源结果表明，edge惩罚虽缩小6.61至9.94个百分点，原reflect四个urban×draw单元却下降1.444／.944／.556／.444个百分点，固定候选已关闭。P26旧scratchE30/4L/practical(ordinary)并非本轮P43的精确父，但主要干预假设已有直接负证据；仅换父不足以重开。

## 全部控制条件

下表是源数据准确率，不是正式测试成绩。每个satellite单元为3600次观察、1800个物理样本，合并两个固定draw；clean每模型1800。

|条件|AS lowSuburban %|AS midUrban %|AS lowUrban %|mid变化 pp|low变化 pp|
|---|---:|---:|---:|---:|---:|
|full|88.666667|72.805556|59.083333|0|0|
|no_noise/fixed|91.416667|78.083333|68.138889|5.277778|9.055556|
|no_noise/total|91.472222|78.361111|70.166667|5.555556|11.083333|
|flat_H/fixed|89.833333|76.166667|60.722222|3.361111|1.638889|
|flat_H/total|89.944444|76.111111|60.916667|3.305556|1.833333|
|no_carrier/fixed|88.944444|72.888889|59.388889|0.083333|0.305556|
|no_carrier/total|88.944444|72.888889|59.416667|0.083333|0.333333|
|no_RX_IQ/fixed|88.666667|72.805556|59.111111|0.000000|0.027778|
|no_RX_IQ/total|88.666667|72.805556|59.083333|0.000000|0.000000|
|actual_power_quality|88.666667|72.444444|59.111111|-0.361111|0.027778|
|edge_boundary|74.694444|61.916667|51.083333|-10.888889|-8.000000|

|固定full条件|NOAS30准确率 %|AS30准确率 %|As−NoAs pp|NOAS宏F1 %|AS宏F1 %|
|---|---:|---:|---:|---:|---:|
|clean|98.444444|98.333333|-0.111111|98.441469|98.329950|
|practical_low_suburban|49.861111|88.666667|38.805556|46.937084|88.571599|
|practical_mid_urban|43.555556|72.805556|29.250000|41.595346|72.719826|
|practical_low_urban|35.611111|59.083333|23.472222|34.525219|58.744394|

As在去噪条件下仍保持明显优势；在low urban的no_noise/fixed，NoAs为41.333333%、As为68.138889%，说明基础As收益不能简单概括为降低新增噪声敏感性。本轮固定两个模型，以上不是新增训练取得的收益。

## 分层、接收状态与纠错

|AS源分层|n|full %|固定状态去噪 %|重算状态去噪 %|
|---|---:|---:|---:|---:|
|mid BLOCKED|1334|36.957|49.925|50.600|
|mid unlocked|680|31.471|45.882|46.324|
|mid <3|383|27.415|44.648|44.386|
|mid 3-8|718|38.579|50.557|52.228|
|low BLOCKED|1979|32.794|47.297|51.036|
|low unlocked|1566|30.332|45.913|49.808|
|low <3|1127|27.329|44.809|48.270|
|low 3-8|794|39.673|50.504|54.660|

上述固定状态去噪在两draw都为正，但最困难组去噪后仍约44%至55%。同一个物理包可同时出现在BLOCKED、unlocked与噪声比分组，不把分组净数相加。新增噪声比不是原录音总SNR。low flat_H/fixed在unlocked与<3dB组跨draw换号，不强称多径单因已被确定。

固定去噪的mid修复328／新增错误138／净+190，low为539／213／+326。重算状态相对固定去噪，mid仅多0.277778个百分点（110修复／100新增），两draw分别−.111111／+.666667；low多2.027778（263／190），两draw均正。总效应包含锁定、补偿和AGC反馈，不能冒充固定状态直接效应。

edge控制mid为139修复／531新增／净−392，其中>=15dB组净−303，约占77%；low为186／474／−288，其中>=8dB两组净−216，占75%。这是改变人工前缀后的受控输入效应，未更改正式receivedIQ或推断真实在轨损伤。

本轮固定NoAs30 clean正确集合1772包、As自身clean正确集合1770包，均非E20父集合。As自身集合mid保持率full73.757062%→固定去噪79.039548%→total79.237288%；low59.745763%→69.011299%→70.988701%。完整RX/day/TX、state、lock、state×lock与新增噪声比的accuracy、macroF1、CM、margin÷30、两种保持率、配对错误和交互见CSV。新读出未覆盖完整V27000或源high urban，不用部分面板冒充全源或全部六正式场景。

## 现有分支、静态门控与梯度

仅复用真实前向钩子，无诊断分类器。mid/low full相对clean的分支平均cosine：time .928071→.939203、frequency .970790→.977111、PA .845074→.884689、reference .279370→.293826、signature .126883→.423113（NoAs→As）。这表明已有As训练改变了可用证据，但特征接近不是识别收益替代指标，不能据此再强制所有分支对齐。

每模型同90个源包的clean与六个full条件，测的是冻结eval-mode smoothedCE局部梯度，不是训练时随机RDrop/SupCon梯度。whole梯度cosine六次平均NoAs约−.294、As约+.206；所有梯度有限且未累积.grad，权重、buffer与eval/gradflags保持。默认头和各分支梯度逐项、960个既有静态坐标gain保留在表中，不编造成逐包门控。尚不能从这些局部量证明训练已收敛或clean正式保持。

## 真实执行基线与权限

P43原B00/B01各自从头E20，再自身student／EMA追加E10，最终用student内联合训练默认头；本轮仅严格加载其model状态。共读取4次ownE20／FINAL30 CP，EMA字典可能随CP反序列化但不应用。实际source角色、scratch祖先、final_only与无target训练/选模核实；无其他父、无P78权重。

完整L6300/U56700隐藏/V27000，源RX1/3/4/6/8、day1/2/3，split392005；无Ldev或源RX留出。只有固定V1800参与诊断；源cache可能反序列化未使用L，U真值始终未读。网络F/SIG_TIME_X3/统一Sinc/joint160/defaultcos6×160 scale30，900722参数／206状态；没有替换、拟合或冻结后另训头。

实际模块__file__确认leo_practical来自既有r02的experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py，DataAugmentation/SSDG同来自该真实运行目录，不是顶层code。现有源面板actual_engine实读：IQ256、fs25MHz、fc2.45GHz、refSNR30、residual/post_sync、EQfalse即G1、quality adaptation/phase tracking/IQ compensation/AGC均true、AGC limit30dB、reflect、fractional_delay_half_length24。它是已缓存源观测实际值；P43训练载频2.462GHz另列，不把两者混同。RMS center=false、eps1e-12的位置沿旧原生数据流程，前后未改。

## 成本、验证与产物

新增NN242464=source prediction241200+eval梯度前向1260+真实CP smoke4；新增训练轮、optimizer/EMA/headfit更新、新信道draw、renderer调用均0。本地真实NN/CP/IQ0。两个既有父各E30／6660更新／native7960684NN，仅复用不重新收费；不能把0新训练写成没有父成本。

源probe实际墙钟31.294900302秒；分析提交至complete为23.223122597秒，新增NN/IQ/CP读取均0；独立mtime观测从probe启动至分析完成经过105.893837690秒，其中包含owner阶段间等待。两个所属进程已缺失，源码/源预测/分析全部完成；单seed不能提供跨seed波动。

11项stdlib协议检查PASS；分析8项数值检查PASS，绘图fixture本地缺matplotlib明确skip；实际服务器已成功生成4PNG/4PDF，四PNG已查看、PDF头均%PDF-1.4。唯一持续P0/P1审查修复arm/label接口P1并用真实输出schema定点确认，最终PASS。单归档传输VERIFIED、13py remotecompile PASS；launcher真实2CP/defaulthead/五分支smoke4NN在IQ前PASS并直接继续；所属PID/CWD/argv/实际GPU0/日志增长独立核实。未为证据重复已通过检查。27份聚合文件2798390byte取回并验证，未下载IQ/CP/原始logits。

实际命令（普通账号，CWD=PROJECT/releases/cvs_FcrossRX_current_controls_E30_20261019_r01）：

~~~text
CVS-RFFI/python -u experiments/cvs_controlled_revisit79/runner.py --project PROJECT --device cuda:0
CVS-RFFI/python -u analysis/cvs_current_controls79_analysis_20261019.py --project PROJECT --run-root PROJECT/runs/20261019-diagnostic-FcrossRX-current-controls-e30-s2026092701-m2-r01 --out PROJECT/runs/20261019-diagnostic-FcrossRX-current-controls-e30-s2026092701-m2-r01/analysis_r01
~~~

[固定配置](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_controlled_revisit79/design.py)、[实际读取入口](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_controlled_revisit79/probe.py)、[分析实现](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_current_controls79_analysis_20261019.py)、[执行与验证](E:/type10-7/automation_reports/CV-SincNet/20261019-diagnostic-FcrossRX-current-controls-e30-s2026092701-m2-r01/validation.md)。

- [逐条件指标](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/condition_metrics.csv)／[控制配对效应](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/control_effects.csv)／[总效应减固定效应](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/total_minus_fixed.csv)
- [As−NoAs](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/As_training_effects.csv)／[训练与控制交互](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/training_control_interactions.csv)
- [分支](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/branch_metrics.csv)／[静态gain](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/static_gates.csv)／[eval梯度](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/eval_CE_gradients.csv)
- [控制效果图](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/control_gain.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/control_gain.pdf)
- [源分层图](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/source_strata.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/source_strata.pdf)
- [As训练效果与交互图](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/training_effect.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/training_effect.pdf)
- [分支与局部梯度图](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/branch_gradients.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_current_controls_E30_20261019_r01/retrieved_r01/analysis_r01/branch_gradients.pdf)

## 取舍与未完成

保留原完整As及普通/raw监督；不据本轮重开carrier、RX IQ、全部对齐、EMA、qualityRisk/Strat/boundedRisk/UPractical/公共传播或边界混合的旧固定负路线。P77还已完整检验按实际新增噪声比的连续有界soft-target标签平滑，并有置换质量对应的匹配控制；它并非CE标量重权，仅换父不能称新机制。

补充事实核查已完成，真实prehistory候选也关闭：既有ManySig.pkl只有576个(1000,256,2) ndarray形状元数据，每条原生sig_i索引对应256点记录；576个4096000byte数值载荷合计2359296000byte均通过seek跳过。没有执行pickle反序列化或构造数值数组，实际读取的是形状/类型等元数据，不把文件体积推断当证据。该数组形状与6TX×12RX×4day×2eq的576组以及native逐记录索引一致；当前供给文件没有裁剪前的更长记录，1000是packet轴，不证明相邻sig_i时间连续，不能拼出真实前历史。原目录也没有连续offset等sidecar。只读工具提交 9c5badb0cf9c22b923c1b311ae7db00fd8039a1f，协议2/4/5的256与600点合成元数据检查PASS，畸形输入拒绝；该工具仅辅助事实核查，非数据重验证或VALIDATED_ONCE收据。没有扩大数据权限、读取数值波形或启动新训练。

本轮验证的是当前源基座的退化机制与一项上下文可行性，尚无新的源证据支持立即重开已失败训练配方。下一项实际训练仍需独立、与旧负结果有关键差异的假设；不会用当前已知正式成绩选择性重跑。

本轮没有训练新模型或新增正式评分，不能更新正式clean/六practical成绩，也没有新增证据证明0.1pp阶段目标、low urban>60或可重复双提升。正式逐包state因果与跨模型seed仍未证明；目标保持ACTIVE。