"""Build the report from fully audited local results; no training or network calls."""
import argparse,base64,csv,html,json,re,shutil,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

sys.stdout.reconfigure(encoding='utf-8')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,default=Path('E:/type10-7/local_artifacts/core90_results_392005'))
parser.add_argument('--output',type=Path)
args=parser.parse_args()
P=args.input
ROOT=Path('E:/type10-7')
WT=ROOT/'code/snapshots/core90_evidence_20260911_wt'
REL=Path('automation_reports/CV-SincNet/core90_evidence_frozen_392005_20260911/comprehensive_report')
OUT=args.output or ROOT/REL
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)
(OUT/'data').mkdir(exist_ok=True)
A=json.loads((P/'comprehensive/comprehensive_analysis.json').read_text(encoding='utf-8'))
S=json.loads((P/'summary.json').read_text(encoding='utf-8'))
V=json.loads((P/'source_summary.json').read_text(encoding='utf-8'))
M=list(S['results']);SC=list(S['results']['H0']['scenes'])
NAMES=['clean','LEO晴空弱','LEO低仰角弱','LEO雨衰弱']
for f in (P/'comprehensive').glob('*'):
    if f.suffix in ('.json','.csv'):shutil.copy2(f,OUT/'data'/f.name)
for name in ['metrics.csv','grouped_metrics.csv','summary.json','source_summary.json']:
    shutil.copy2(P/name,OUT/'data'/name)
for name in ['plan.json','data_summary.json','all_source_frozen.json','status.json','commands.jsonl']:
    shutil.copy2(P/'evidence'/name,OUT/'data'/name)
contract=json.loads((P/'evidence/data_contract.json').read_text(encoding='utf-8'))
compact={k:v for k,v in contract.items() if k!='roles'}
compact['role_counts']={k:len(v) for k,v in contract['roles'].items()}
(OUT/'data/data_contract_metadata.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'data/H0_runtime_args.json').write_text(json.dumps(A['H0_baseline_args'],ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(WT/'code/configs/core90_evidence_h0_h5.json',OUT/'data/core90_evidence_h0_h5.json')
H=list(csv.DictReader((P/'comprehensive/H0_training_history.csv').open(encoding='utf-8-sig')))
HH=list(csv.DictReader((P/'comprehensive/head_training_history.csv').open(encoding='utf-8-sig')))
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':180})
def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight')
    svg=OUT/'figures'/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
vals=np.array([[S['results'][m]['scenes'][s]['overall']['closed_set_accuracy']*100 for s in SC] for m in M])
fig,ax=plt.subplots(figsize=(8.6,5.6));im=ax.imshow(vals,cmap='YlGnBu',vmin=30,vmax=80)
ax.set_xticks(range(4),NAMES);ax.set_yticks(range(8),M)
for i in range(8):
    for j in range(4):ax.text(j,i,f'{vals[i,j]:.2f}',ha='center',va='center',color='white' if vals[i,j]>62 else '#15273c')
fig.colorbar(im,ax=ax,label='闭集准确率（%）');ax.set_title('同一冻结骨干：8候选×4场景\n每格168000条；单seed392005',pad=14);save(fig,'accuracy_matrix')
fig,axs=plt.subplots(2,2,figsize=(12,7.5));ep=[int(x['epoch']) for x in H]
axs[0,0].plot(ep,[float(x['train_loss']) for x in H],color='#225ea8');axs[0,0].set_ylabel('H0总损失')
axs[0,1].plot(ep,[float(x['val_tx_acc']) for x in H],color='#117864');axs[0,1].set_ylabel('source V准确率（%）')
axs[1,0].plot(ep,[float(x['train_loss_sat_cls_labeled']) for x in H],label='LEO分类损失');axs[1,0].plot(ep,[float(x['train_loss_unlabeled']) for x in H],label='U_s损失');axs[1,0].legend();axs[1,0].set_ylabel('记录的损失项')
axs[1,1].plot(ep,[float(x['train_pseudo_selected']) for x in H],color='#a65e00');axs[1,1].set_ylabel('伪标签接收事件数/epoch')
for ax in axs.flat:
    ax.axvline(80,color='#aaa',ls='--',lw=1);ax.axvline(131,color='#aaa',ls=':',lw=1);ax.set_xlabel('epoch');ax.grid(alpha=.15)
fig.suptitle('H0完整200轮：E80启动LEO分类损失，E131启动U_s伪标签');fig.tight_layout();save(fig,'training')
fig,axs=plt.subplots(1,2,figsize=(12,4.6))
for m in ['H0','H1','H2','H3','H4','cosine','linear_ridge']:
    pts=A['target'][m]['leo_low_elev_weak']['risk_at_coverage'].values()
    axs[0].plot([x['attained_coverage']*100 for x in pts],[x['risk']*100 for x in pts],marker='.',label=m)
for m in ['H0','H1','H2','H3','H4','cosine']:
    rows=list(csv.DictReader((P/'comprehensive/reliability_bins.csv').open(encoding='utf-8-sig')))
    rr=[r for r in rows if r['method']==m and r['scene']=='leo_low_elev_weak' and int(r['count'])>0]
    # Schema aliases kept explicit below after reading the exported header.
    ck='mean_confidence';ak='accuracy'
    if rr:axs[1].plot([float(r[ck]) for r in rr],[float(r[ak]) for r in rr],marker='.',label=m)
axs[0].set(xlabel='实际置信度覆盖率（%）',ylabel='接收风险（%）',title='低仰角LEO：置信度选择诊断点');axs[0].legend(ncol=2,fontsize=8)
axs[1].plot([0,1],[0,1],'--',color='#999');axs[1].set(xlabel='平均置信度',ylabel='实际正确率',title='低仰角LEO：15等宽区间可靠性');axs[1].legend(ncol=2,fontsize=8)
fig.tight_layout();save(fig,'risk_and_calibration')
fig,axs=plt.subplots(2,3,figsize=(12,8))
for ax,m in zip(axs.flat,['H0','H1','H2','H3','H4','cosine']):
    cm=np.array(A['target'][m]['leo_low_elev_weak']['closed_set_confusion'])/28000*100
    ax.imshow(cm,vmin=0,vmax=100,cmap='Blues');ax.set_title(m);ax.set_xticks(range(6));ax.set_yticks(range(6));ax.set_xlabel('预测类');ax.set_ylabel('真实类')
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{cm[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if cm[i,j]>55 else '#14283f')
fig.suptitle('低仰角LEO混淆矩阵：按真实类归一化（%），每类28000条');fig.tight_layout();save(fig,'confusion_low_leo')

L=[]
def prose(s):L.extend([s.strip(),''])
def table(headers,rows):
    L.extend(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|'])
    L.extend('|'+'|'.join(str(c) for c in r)+'|' for r in rows);L.append('')
def fig(name,caption):prose(f'![{caption}](figures/{name}.png)\n\n{caption}。提供同名SVG用于编辑和出版排版。')
def pct(x):return f'{x*100:.3f}'
prose('''# CORE90部分证据与条件响应头：全面实验报告

实验：core90_evidence_frozen_392005_20260911｜seed=392005｜报告日期：2026-09-11

**完成状态：VERIFIED，8候选×4场景全部预测与独立评分完成。**本报告覆盖方法机制、设计追踪、实际训练激活、数据与checkpoint契约、完整目标指标、分组结果、错误结构、拒识与校准、稳定性和资源。H0完整训练200轮，H1–H4各完成20轮冻结骨干拟合；目标每场景168000条，共32个预测结果。目标数据跨场景重复，不能把5376000条“预测行”当作相互独立样本。

**主要结论：新证据头本轮没有改善闭集跨接收机识别。**H1–H4四场景全部低于H0；低秩H2比对角H1好，但仍明显低于H0；加入条件响应的H3进一步下降，H4恢复部分性能，未超过H2。cosine只取得0.105–0.371个百分点的小幅描述性增益。部分概率校准指标改善，但校准处理不匹配且拒识覆盖率下降，不能据此宣布综合性能提升。

这是固定6类TX的Phase1研究，结论限定于ManySig均衡IQ、当前源/目标接收机划分和合成LEO弱场景。它不是新TX注册或target K-shot实验，也未证明真实发射激励、PA指纹解耦、未知类发现或真实在轨泛化。只有一个训练seed，结果不用于晋级，不回流目标调参、选模或补跑。''')
prose('[TOC]')
prose('## 1.研究问题与已执行范围')
prose('''原设计希望把“证据是否被观测到”“观测误差有多大”“证据之间是否相关”“状态变化怎样影响类响应”放入统一概率模型，避免把缺失当成零、把相关证据重复累加，或把接收机增益误解释为发射端激励。

本轮执行已接受的第一阶段：按当前数据契约从零训练H0，固定E200骨干，然后在同一特征空间比较H1–H4及三个常规头。骨干冻结隔离了头部效应，也意味着新概率头不能反向改善特征。H5类对专家、480维分块布局、联合训练和目标support注册属于未执行分支；其代码或单元测试存在不等于本轮获得了科学结果。''')
table(['候选','本轮实际机制','训练/拟合范围'],[
['H0','原CORE90判别头','完整骨干与原头从零训练200轮，固定E200'],
['H1','160维joint证据＋对角类内协方差＋source受控退化观测误差','冻结骨干；20轮；状态响应关闭'],
['H2','H1＋rank=4低秩相关协方差','冻结骨干；20轮；状态响应关闭'],
['H3','H2＋二维received-IQ状态条件响应＋状态误差传播','冻结骨干；20轮'],
['H4','H3＋source support posterior episode训练','冻结骨干；20轮；target前向没有support注册'],
['cosine','source类均值方向的余弦分类','同一H0特征，source拟合'],
['diagonal_gaussian','常规类条件对角高斯对照','同一H0特征；不同于含观测模型的H1'],
['linear_ridge','线性岭回归对照','同一H0特征，source拟合'],
['H5','共享反对称类对残差与全局耦合','条件性后续；本轮未启动']])
prose('## 2.方法机制、代码路径与数学含义')
prose('''### 2.0 H0骨干与原训练目标

实际部署是`DualCVSincNetDisentangle`双分支表征，`model_size=M`、`model_variant=lite_d`、`representation_mode=dual`；身份分支读取`feat_joint`，`branch_ablation=no_dac`，域分支`domain_branch_ablation=no_stats`，域增强为`rcn_stats`、强度0.35。源域domain数量15，对应5个source RX×3个source day。新头接在160维身份联合特征上，不能把480维分块设计当成本轮实际输入。

H0保留CORE90的监督TX分类、域分类/对抗、正交与Fishr等约束，以及原型、开放空间、身份紧致、proxy unknown、soft mix、source episode和LEO/伪标签调度；具体权重及起始轮次以配置附件和第4节实际loss为准。后加的其他方法默认被显式关闭。这里的proxy unknown是source训练构造，当前目标固定6类，没有真实未知类别评价。

完整配置附件同时保存历史默认值、历史命令、本次显式协议覆盖及后加功能关闭项。构造优先级为historical defaults→later addon overrides→original command→intentional overrides→实际物理选择器；因此附件中的历史`joint_safe`、0.10/0.70/0.20不能误读为本次有效协议。本次使用scratch、0.07/0.63/0.30、source-only和final_only。`H0_runtime_args.json`只包含部署重建所需的架构参数，不是完整训练超参数清单。''')
prose('''### 2.1观测、缺失与固定缩放

`evidence_observation.py`从骨干aux提取证据。本轮`layout=joint`使用160维`feat_joint`；可选`blocks`拼接`t_emb/f_emb/pa_local`，本轮未用。每块除以固定√块宽，mask变化不会改变保留坐标的数值尺度。缺失坐标由布尔mask定义，不作为实测零送入概率距离。

训练使用0.15随机特征丢弃。对全部32个目标预测张量的复核发现，H1–H4在所有场景、所有168000条样本上都观测到160维。因此本轮验证的是**完整证据目标预测**，并没有目标缺失模式、非随机缺失或缺失强度外推的性能证据。

### 2.2部分高斯、相关性与观测误差

设有效坐标集合为O，类响应为μ_y(e)，总协方差包含正对角项D、低秩项UUᵀ、样本观测误差R(q)和启用时的状态误差传播项。先在O上取协方差子矩阵，再求逆/线性求解；不能先求完整精度矩阵再截取。

```text
C_y,O = [D + U Uᵀ + R(q) + J_y Σ_e J_yᵀ]_(O,O)
r_y,O = z_O - μ_y(e)_O
score_y = -0.5 × (r_y,Oᵀ C_y,O⁻¹ r_y,O + logdet(C_y,O) + |O| log(2π))
```

这是实现的核心概率结构示意；不同消融关闭相应项。`partial_gaussian_head.py`提供部分高斯、正定参数化和低秩求解。H1的rank=0；H2–H4的rank=4。同一信息的相关性通过协方差进入，而不是单独重复计票。技术验收曾对照稠密解和梯度；正式运行中低秩参数非零的证据见第4节。

`evidence_conditions.py`使用L_s受控退化产生误差代理，拟合有上下界的观测方差。其系数冻结，不能让分类CE任意改写质量定义。保存的calibration bias仅作诊断，没有用作预测偏置修正。这里的quality是**source受控退化误差代理**，不是外场真实噪声协方差或实测物理误差界；对非随机缺失也没有识别性保证。

### 2.3条件响应与状态误差传播

`conditional_response.py`用二维received-IQ状态代理驱动类均值：类中心加共享与类特定斜率，状态限制在source覆盖范围，并记录域外状态；通过Jacobian将假定状态误差传播到证据协方差。H1/H2将状态分支作为消融置零；H3/H4开启。

接收IQ的幅度/形态会受到RX、信道和TX共同影响。received proxy不等于发射机功率、输入激励或PA工作点。本轮source探针能够从state预测RX，且高于参考水平，因此不能宣称接收机无关或已辨识物理PA响应。默认状态误差方差是模型设定，不是实测Fisher界。

### 2.4support后验与H4的实际边界

对同类support的部分、异方差观测，条件响应参数用联合高斯条件化形成后验；重复physical ID必须拒绝，多个view不能充当多个shot。query只使用冻结后验进行预测，不更新注册统计。

H4每轮实际执行2952个source episode query，20轮累计59040个query事件。这证明训练路径真正运行，但不是59040个独立新样本。当前Phase1目标入口没有调用target support posterior，故H4结果只检验source episode训练后的冻结头，不检验K-shot注册、未知TX、support覆盖不足或query顺序无关的目标收益。

### 2.5类对诊断与H5

`pairwise_evidence.py`实现类对D²、条件新增证据的Schur补、Fisher适用条件诊断，以及共享反对称残差和全局耦合投影。这些是诊断/建模工具，不保证真实错误率下降。正式run没有保存类对D²、Schur/Fisher数值，H5没有启动；本报告不以技术测试替代正式性能，不构造未测结果。

### 2.6训练目标与source冻结决策

```text
L_head = CE + 0.01 × Gaussian density NLL
         + 0.001 × response parameter regularization
         + 0.1 × support episode CE  [仅H4激活]
```

L_s拟合头及受控退化观测模型；V拟合温度和一致性分组阈值；全部source状态固定后才开始目标预测。决策区分identify、defer、model_mismatch_candidate；后者是模型不匹配候选，不是真实未知类标签。目标所有候选/场景预测固定后，独立scorer才连接truth。H0和三个常规对照没有匹配H1–H4的温度校准流程，NLL/ECE比较是部署系统差异，不能纯归因于高斯结构。''')
prose('### 2.7原设计逐项追踪：实现、运行与科学证据')
table(['ID','要求','本轮证据与剩余边界'],[
['R01','H0原头/快速推理对照','正式H0＋三常规对照完成；H1–H4非证据参数逐tensor与H0完全相同'],
['R02','结构化观测、mask、预处理','代码及训练dropout激活；目标全160维，未测实际缺失泛化'],
['R03','正确部分高斯边缘化','稠密参考技术验证；正式全观测高斯路径运行'],
['R04','方向性误差、偏置/缺失偏差','source退化观测代理激活；物理误差与非随机缺失未验证'],
['R05','低秩相关性','H2–H4末态factor非零；目标效果未超过H0'],
['R06','受约束条件响应','H3/H4斜率非零；state仍含RX关联，物理语义未证实'],
['R07','状态误差传播','H3/H4训练路径运行；默认状态误差是设定'],
['R08','support后验','H4真实source多类episode激活；target K-shot未执行'],
['R09','类对/新增证据/Fisher','代码技术验证；本轮没有正式数值诊断产物'],
['R10','共享类对专家','H5技术实现；按原计划条件性推迟，未执行'],
['R11','损失、校准、歧义与不匹配','source拟合后冻结；目标拒识/风险/覆盖完整评分'],
['R12','配置消费、导出与评分','当前Phase1的8×4预测、seals、独立truth-last评分闭合']])
prose('历史实现追踪保留在[原追踪文档](../../../../analysis/core90_evidence_traceability.md)，其“正式科学评估未开展”描述对应发布前时间点。本报告提供完成后的运行与性能证据，不将历史技术验收改写成物理验证。')
prose('## 3.实验设计、数据权限与复现条件')
table(['项目','实际设置'],[
['数据','ManySig.pkl；均衡IQ，6类TX；input_len=256；采样率25MHz，载频2.462GHz'],
['source RX','1、3、4、6、8；source day=1、2、3'],
['target RX','0、2、5、7、9、10、11；实际target day=0、1、2、3'],
['L_s','6300条＝6×5×3×70；用于监督/头拟合'],
['U_s','56700条＝6×5×3×630；仅source无标签路径'],
['V','27000条＝6×5×3×300；source校准/诊断'],
['target','168000条＝6×7×4×1000；每RX24000，每day42000，每TX28000'],
['初始化','scratch H0；H1–H4只继承本次同契约H0的冻结骨干'],
['固定预算','H0=200轮，batch=128；H1–H4=20轮，batch=64；eval batch=64'],
['seed','392005；不混用此前384407或其他历史run结果'],
['头优化','AdamW，lr=0.001，weight_decay=0.01；无头部早停或target选模'],
['头数据遍历','每轮99批，覆盖6300条L_s；每头1980个优化步'],
['正式选择','固定H0 E200；source冻结完成后统一预测，全部预测后独立评分']])
prose('''source总计90000条按0.07/0.63/0.30划分。target的day0是严格未见日期子集，42000条；day1–3虽日期与source重合，RX仍完全未见。因此总体168000条应称跨RX混合日期测试，不能全部称严格跨RX且跨日期。第7节分别给出两组。

正式checkpoint检查覆盖数据角色、物理ID和继承来源。本轮H0从零初始化，H1–H4继承当前H0，避开旧CORE90曾接触目标的权重。部署状态复核确认所有非证据骨干tensor逐位相同。这证明公平冻结与当前继承关系，不替代对未来checkpoint来源的检查。

当前结果使用同一批物理ID。32份预测全部读入、逐项验证ID集合和顺序与truth一致、score有限、重新计算正确数与原评分匹配。完整物理映射保存在原始data_contract.json，报告附不含长ID列表的metadata；不重复执行数据builder验证。''')
prose('类别编号0–5的物理TX映射依次为：`14-10`、`14-7`、`20-15`、`20-19`、`6-15`、`8-20`。混淆矩阵和TX分组表均使用该固定顺序。')
prose('### 3.1合成LEO场景与物理解释边界')
table(['场景','仰角°','SNR dB','CFO标准差Hz','相位噪声增量标准差范围','K因子dB','衰落/最大延迟采样数/功率衰减/AGC范围'],[
['clean','—','原始均衡IQ','—','—','—','不叠加此处LEO代理'],
['leo_clear_weak','35–90','22–32','50','0–0.0005','16–24','Rician；delay=2；decay=0.08；AGC±0.2dB'],
['leo_low_elev_weak','10–35','16–28','90','0.0001–0.0008','8–18','shadowed Rician；delay=3；decay=0.12；AGC±0.3dB'],
['leo_rain_weak','20–80','14–26','70','0.0001–0.0007','10–20','Rician；delay=3；decay=0.10；AGC±0.3dB']])
prose('这些是`leo_residual`简化残余信道预设，2 taps；真实路径损耗、大气衰减模型和IQ不平衡开关关闭。“rain”是预设名，不表示已完整模拟物理雨衰链路；没有真实卫星采集或在轨验证。精确参数以发布代码`training_controls.py`与`sat_channel.py`及runtime args为准。')
prose('## 4.实际激活、收敛和数值稳定性')
fig('training','图1：完整H0训练轨迹，不以末尾抽样代替全量日志')
prose('''H0总损失从16.930914降至6.437092。E79到E80的损失上升伴随LEO分类损失首次非零，是调度切换，不能直接诊断为发散。训练阶段为S1 E1–16、S2 E17–68、S3 E69–130、SSDG pseudo E131–200。source V最高98.203704%@E172，但本轮按预设使用E200，训练记录的E200 V为97.859259%。冻结导出后重新评估V为97.862963%，相差1/27000条；两者是不同评估调用，不混成同一测量，具体数值差异原因未证实。

LEO增强配置在E1即存在，实际LEO分类loss于E80开启（λ=0.68）；卫星一致性loss全200轮为0。U_s loss于E131首次非零，累计处理439040个采样事件、接收2854个伪标签事件，接收率约0.65006%；不是56700条U_s每轮全部有效参与，也不是2854个独立物理样本。E200接收27/6272，U_s loss仅2.066×10⁻⁷：路径激活但利用率较低，不能只凭开关宣称强贡献。''')
table(['H0损失项','非零轮数','首次非零epoch','E200值'],[[x['metric'],x['nonzero_epochs'],x['first_nonzero_epoch'],f"{x['last']:.6g}"] for x in A['H0_mechanisms']])
prose('### 4.1冻结证据头参数与真实路径')
table(['方法','可训练参数','rank','factor平方范数','response斜率平方范数','最终对角方差范围','拟合分钟'],[[m,h['head_trainable_count'],h['config']['configured']['covariance_rank'],f"{h['factor_squared_norm']:.6g}",f"{h['slopes_squared_norm']:.6g}",f"{h['diag_variance_min']:.6g}–{h['diag_variance_max']:.6g}",f"{h['fit_seconds']/60:.2f}"] for m,h in A['heads'].items()])
table(['方法','首轮总loss','末轮总loss','末轮CE重建','末轮NLL/批','末轮support CE/批','实际support query事件/20轮'],[[m]+[f"{float(x):.6f}" for x in [next(r for r in HH if r['method']==m)['loss'],[r for r in HH if r['method']==m][-1]['loss'],[r for r in HH if r['method']==m][-1]['ce_reconstructed'],[r for r in HH if r['method']==m][-1]['evidence/nll_batch_mean'],[r for r in HH if r['method']==m][-1]['evidence/support_loss_batch_mean']]]+[int(sum(float(r['evidence/support_queries']) for r in HH if r['method']==m))] for m in A['heads']])
prose('''原始head history中的`evidence/*`多数为99批的累加值，报告的“/批”除以99；不能把84左右的observed_fraction累加值解释为8400%观测。平均观测约85%，符合0.15训练dropout。总loss允许为负，因为固定尺度的连续高斯密度NLL可以为负；它不是分类NLL，也不等同于非有限异常。H4末轮support CE约4.4445，虽然source完整视图V很高，episode目标仍较难。

逐个部署检查发现H1–H4所有非证据tensor与H0逐位相等。H2相关项、H3/H4响应斜率均非零，支持相应训练路径已更新；但非零参数不自动证明性能贡献。消融结果显示这些结构在本轮跨域条件下仍可能损害识别。

### 4.2异常审计

完整200轮H0、4×20轮head history和全部17份可用日志均已解析。H0非有限loss跳步为0，非有限梯度跳步共10/9800≈0.10204%，发生于E1、6、28、67、108、116、133、146、174，其中E6两次；训练最终完成。日志中未发现Traceback或OOM。未启用诊断及训练期target指标中的null/NaN不作为训练loss异常计数；完整字段审计见附带JSON。没有对报告编写重新运行GPU训练。''')
prose('## 5.完整目标总体指标与消融解释')
fig('accuracy_matrix','图2：8×4闭集准确率总览')
table(['方法']+NAMES+['三LEO平均%'],[[m]+[f'{x:.3f}' for x in vals[i]]+[f'{vals[i,1:].mean():.3f}'] for i,m in enumerate(M)])
prose('''H2相对H1的改善说明低秩相关结构在当前冻结特征上比纯对角头更合适；H3相对H2下降，说明本轮状态条件响应没有带来泛化优势。H4相对H3恢复部分准确率，但仍未追上H2或H0。这些是相邻消融的观察关联，不能脱离训练目标、优化和状态混杂解释为已确定的因果机制。

H1–H4的source V都接近98%，目标LEO却下降到约46%–53%，表明source拟合充分不能保证目标转移。当前证据不支持增加结构复杂度就会提高跨RX鲁棒性。cosine的增益很小且单seed，不构成可推广的胜出结论。''')
prose('### 5.1指标定义')
prose('''闭集准确率=全部样本argmax正确数/N；有效准确率=正确且被接收数/N，拒绝计错；覆盖率=接收数/N；接收风险=接收错误数/接收数。NLL、Brier和15等宽bin ECE对全部样本计算，越低越好。最弱RX与最弱RX×TX都是对应分组的闭集准确率下界，不是置信区间。百分数和百分点明确区分。''')
for s,n in zip(SC,NAMES):
    prose('### 5.'+str(SC.index(s)+2)+' '+n+'：全部部署指标')
    rows=[]
    for m in M:
        r=S['results'][m]['scenes'][s];o=r['overall']
        rows.append([m]+[pct(o[k]) for k in ['closed_set_accuracy','accuracy','coverage','accepted_risk']]+[f'{o[k]:.5f}' for k in ['nll','brier','ece']]+[pct(r['receiver_closed_set_accuracy_floor']),pct(r['receiver_class_closed_set_accuracy_floor'])])
    table(['方法','闭集%','有效%','覆盖%','接收风险%','NLL','Brier','ECE','最弱RX%','最弱RX×TX%'],rows)
prose('## 6.拒识、概率校准与公平覆盖率比较')
fig('risk_and_calibration','图3：低仰角LEO的置信度选择与校准诊断')
prose('''低仰角LEO中，H2闭集正确86283条；其中接收正确68864、接收错误28913、拒绝正确17419、拒绝错误52804。其有效准确率40.990%、覆盖率58.201%、接收风险29.570%。相比H0的100%覆盖、42.207%风险，风险下降同时移除了大量样本，包括17419条本来可判对的样本。因此不能用29.570%直接宣称优于H0。

下表统一按最大softmax置信度作选择，按相同置信度整组保留，不用truth拆分ties，不对达不到的覆盖率插值。它是**离线置信度诊断策略**，未包含H1–H4原部署的一致性/不匹配门控，不能替代原部署结果，也不将目标阈值回写部署。''')
table(['方法','请求50%时实际覆盖%','风险%','请求80%时实际覆盖%','风险%','tie-group右矩形AURC'],[[m,pct(A['target'][m]['leo_low_elev_weak']['risk_at_coverage']['0.5']['attained_coverage']),pct(A['target'][m]['leo_low_elev_weak']['risk_at_coverage']['0.5']['risk']),pct(A['target'][m]['leo_low_elev_weak']['risk_at_coverage']['0.8']['attained_coverage']),pct(A['target'][m]['leo_low_elev_weak']['risk_at_coverage']['0.8']['risk']),f"{A['target'][m]['leo_low_elev_weak']['tie_grouped_right_rectangle_aurc']:.6f}"] for m in M])
prose('''在约80%覆盖时，H0风险36.226%，H2为40.755%；cosine为35.406%。H2未显示优于H0的置信度排序能力。常规diagonal_gaussian约95.6%样本softmax饱和为1，不能声称获得50%或80%的可实现覆盖点。AURC使用按tie整组的右矩形积分，数值依赖该明确约定，不宣称无偏风险认证。

H1–H4的温度来自V，H0/常规对照未进行匹配校准，故其NLL/ECE下降包含校准差异；现有目标结果不足以拆分“高斯结构”和“温度校准”的各自贡献。可靠性图的点不按bin样本量加权展示，定量ECE以表格和CSV为准。''')
prose('## 7.全部RX、日期与TX分组结果')
prose('以下全部为闭集准确率（%）；每场景同一168000条。RX×day、RX×TX的完整2784个分组行及其有效准确率、覆盖率、风险和校准指标见[grouped_metrics.csv](data/grouped_metrics.csv)。极低的单RX×TX不能被平均值掩盖。')
for s,n in zip(SC,NAMES):
    prose('### '+n)
    for key,label in [('by_receiver','目标接收机'),('by_day','日期'),('by_class','TX类别')]:
        groups=list(S['results']['H0']['scenes'][s][key])
        prose('**'+label+'**')
        table(['方法']+groups,[[m]+[pct(S['results'][m]['scenes'][s][key][g]['closed_set_accuracy']) for g in groups] for m in M])
    table(['方法','未见RX＋已见day1–3%（126000条）','未见RX＋未见day0%（42000条）'],[[m,pct(A['target'][m][s]['seen_days_1_2_3']['accuracy']),pct(A['target'][m][s]['unseen_day_0']['accuracy'])] for m in M])
prose('## 8.逐样本配对变化与错误结构')
prose('配对统计对相同物理样本比较：rescue是H0错而候选对；harm是H0对而候选错；净正确变化=rescue−harm。预测改变数还包括“双方都错但错到不同类”，不能等同于rescue＋harm。raw score/margin尺度跨头不同，不作为统一效果量。')
for s,n in zip(SC,NAMES):
    prose('### '+n+'相对H0')
    table(['方法','rescue数','harm数','净正确变化','准确率差/百分点'],[[m,r['rescue_count'],r['harm_count'],r['rescue_count']-r['harm_count'],f"{(r['rescue_count']-r['harm_count'])/1680:.3f}"] for m,r in S['paired_to_H0'][s].items()])
fig('confusion_low_leo','图4：低仰角LEO的主要混淆模式')
prose('低仰角H2纠正5203条H0错误，却破坏16013条H0正确判断，净减少10810条。错误并非仅是低置信样本拒绝问题：闭集决策本身发生了净损失。全部32个6×6原始计数矩阵见[confusion_matrices.csv](data/confusion_matrices.csv)，图中行归一化仅用于阅读；每格原始计数可独立汇总回整体准确率。')
prose('## 9.source校准、状态/质量关联与机制限制')
table(['方法','raw V准确率%','校准后有效%','校准后覆盖%','校准后NLL','校准后ECE','温度'],[[m,pct(v['raw_head']['closed_set_accuracy']),pct(v.get('calibrated_head',v['raw_head'])['accuracy']),pct(v.get('calibrated_head',v['raw_head'])['coverage']),f"{v.get('calibrated_head',v['raw_head'])['nll']:.5f}",f"{v.get('calibrated_head',v['raw_head'])['ece']:.5f}",f"{v.get('calibration',{}).get('temperature',1):.6f}"] for m,v in V.items()])
table(['方法','state→TX%','state→RX%','quality→TX%','quality→RX%'],[[m]+[pct(v['condition_probes'][k]['accuracy']) for k in ['state_TX','state_RX','quality_TX','quality_RX']] for m,v in V.items() if 'state_TX' in v.get('condition_probes',{})])
prose('''TX多数类参考16.667%，RX参考20%。H3/H4的state→RX为43.978%，quality→RX为44.956%；两者明显携带采集关联。state→TX约25.770%、quality→TX约29.263%，也不能当作纯外生状态。H1/H2状态探针接近参考值是状态分支被消融置零的结果，不能据此证明原始状态不含RX信息。全观测mask恒定，所以mask探针参考水平没有非随机缺失独立性的证明力。

H1–H4校准包含3个160维质量分组，最低组样本要求20，分位数0.95，置信门槛0.5；H3/H4 source状态域内比例约99.893%。V同时承担校准和这里的诊断，因而这些source风险不是独立留出认证。source覆盖好也不推出target状态分布合规。''')
prose('## 10.计算成本、运行时序与交付状态')
table(['阶段','实际资源/耗时','解释'],[
['H0完整训练','30877.553秒＝8.577小时；1,049,827可训练参数','含该阶段source评估/报告；epoch合计30827.571秒'],
['H0峰值显存','allocated约9.509GiB；reserved约9.939GiB','实际训练峰值；不是启动时显存快照'],
['H1–H4拟合','18.39/13.50/17.28/41.47分钟','并行任务；不能相加当墙钟时间'],
['head显存','UNKNOWN：没有保存正式head训练峰值','不使用旧synthetic测试显存充当正式结果'],
['总流程','约01:46启动，10:21 H0完成，11:04全部source冻结，12:09评分完成','约10小时23分钟；均为2026-09-11香港时间'],
['结束状态','ARTIFACTS_COMPLETE；监控automation已暂停','本报告没有重启或调优实验']])
table(['候选','四场景预测分钟','预测产物MiB'],[[m,f"{r['resources']['total_prediction_seconds']/60:.2f}",f"{r['resources']['total_artifact_bytes']/1024**2:.2f}"] for m,r in S['results'].items()])
prose('预测耗时包括IQ读取、逐样本信道变换、批量前向和保存，受共享GPU负载影响，不是独占模型kernel延迟。H4拟合成本最高，而闭集效果低于H2；当前结果不支持其综合成本收益占优。没有正式独占吞吐、延迟分位数或head显存数据。')
prose('## 11.科学判定与下一步边界')
table(['问题','当前判定'],[
['是否完成授权冻结矩阵','是：H0 E200＋H1–H4各20轮，8×4结果闭合'],
['方法是否只停留在配置','否：观测模型、相关项、响应斜率、H4 source episode均有实际路径/末态证据'],
['是否提升跨RX闭集精度','否：H1–H4全部场景低于H0'],
['是否获得更好的校准','部分部署NLL/ECE改善；校准不匹配，不能纯归因于结构'],
['是否获得更优选择性识别','当前低仰角等覆盖诊断不支持H2优于H0'],
['是否证明物理条件响应/PA解耦','否：received代理仍关联RX/TX，缺少物理标定'],
['是否证明缺失证据或target K-shot收益','否：目标全160维，未执行target support注册'],
['能否晋级或宣布方法胜出','不能：单seed描述性结果且新头整体退化']])
prose('''后续设计应优先在source允许的信息上辨别状态混杂、概率尺度、低秩建模及目标不匹配风险，而不是依据本次目标分数继续搜索参数。若另行授权新研究，应预先固定校准匹配的对照、缺失机制、support覆盖诊断和独立多seed评估；新的目标反馈必须遵守项目协议。本报告不启动这些后续实验。

当前最可信的交付是一个已验证执行、可复核失败与收益边界的冻结头对比，而不是已成立的物理机制或性能提升论证。''')
prose('## 12.数据附件、复核口径与代码索引')
table(['附件','内容'],[
['[metrics.csv](data/metrics.csv)','32行部署指标、资源和配对补充字段'],
['[grouped_metrics.csv](data/grouped_metrics.csv)','2784行RX/day/TX及交叉分组指标'],
['[confusion_matrices.csv](data/confusion_matrices.csv)','32×6×6＝1152行原始混淆计数'],
['[decision_counts.csv](data/decision_counts.csv)','接收/拒绝与正确/错误四格计数'],
['[risk_coverage_points.csv](data/risk_coverage_points.csv)','32×11覆盖率诊断点，含实际覆盖率和阈值'],
['[reliability_bins.csv](data/reliability_bins.csv)','32×15校准bin，含样本数'],
['[H0_training_history.csv](data/H0_training_history.csv)','完整200轮训练、调度和伪标签记录'],
['[head_training_history.csv](data/head_training_history.csv)','4×20轮头训练，原始累加值与批均值'],
['[H0_mechanism_activation.csv](data/H0_mechanism_activation.csv)','16项loss首次激活/非零轮数'],
['[source_summary.json](data/source_summary.json)','source校准、探针及相关诊断'],
['[comprehensive_analysis.json](data/comprehensive_analysis.json)','完整复核、末态参数、异常审计、日期分解和风险数据'],
['[H0_runtime_args.json](data/H0_runtime_args.json)','H0部署重建的架构参数子集'],
['[core90_evidence_h0_h5.json](data/core90_evidence_h0_h5.json)','完整历史默认、命令、本次协议覆盖和H0–H5配置；按正文优先级解释'],
['[data_contract_metadata.json](data/data_contract_metadata.json)','数据schema、物理映射元数据及角色计数'],
['[plan.json](data/plan.json)','当前seed、预算与调度计划'],
['[commands.jsonl](data/commands.jsonl)','实际阶段命令'],
['[summary.json](data/summary.json)','原独立评分的紧凑全量汇总']])
prose('''原始大文件保留在N607：`/home/szu2070436088/2510044040/CV-SincNet/runs/core90_evidence_frozen_392005_20260911`；本机完整审计缓存：`E:/type10-7/local_artifacts/core90_results_392005`。原comparison.json约587MiB，全部预测张量、truth和日志均已实际读取；报告附件保留紧凑可审阅数据，不将大预测张量重复提交Git。

正式执行代码release：`edca15a6`；此前结果报告提交：`46bbd8fbc467e4176a05d309ac07b47ce3b485c3`。本报告由当前完成产物生成。发布前技术验收144 passed、0 skipped属于既有实现验证，不是本次新增实验样本或新增GPU测试。读者可先核对各confusion对角和、decision四格计数、各分组权重汇总，再对照完整评分。

主要生产代码：''')
for f in ['evidence_observation.py','partial_gaussian_head.py','evidence_conditions.py','conditional_response.py','pairwise_evidence.py','evidence_head_training.py','evidence_decision.py']:
    prose(f'- [{f}](../../../../code/cvsrffi/{f})')
prose('实际入口和配置见[实现报告](../../../../analysis/core90_evidence_implementation_20260911.md)及本目录commands.jsonl。原设计文本与本次范围的对应关系见第2.7节，技术实现、正式运行和科学验证三层证据分别陈述。')
text='\n'.join(L)
# Local delivered copy and mirrored report both point to the verified source checkout.
text=text.replace('](../../../../',']('+WT.as_posix()+'/')
assert '\ufffd' not in text
(OUT/'report.md').write_text(text,encoding='utf-8')
body=markdown.markdown(text,extensions=['tables','fenced_code','toc'])
for name in ['accuracy_matrix','training','risk_and_calibration','confusion_low_leo']:
    b64=base64.b64encode((OUT/'figures'/f'{name}.png').read_bytes()).decode()
    body=body.replace(f'src="figures/{name}.png"',f'src="data:image/png;base64,{b64}"')
doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CORE90全面实验报告</title><style>
:root{color-scheme:light}body{margin:0;background:#f2f5f8;color:#183047;font:16px/1.85 "Microsoft YaHei",sans-serif}main{max-width:1250px;margin:32px auto;padding:48px 60px;background:white;box-shadow:0 4px 24px #16374a0c}h1{font-size:32px;line-height:1.45;border-bottom:5px solid #187b85;padding-bottom:24px}h2{margin-top:58px;color:#116775;border-left:5px solid #187b85;padding-left:15px}h3{margin-top:32px}a{color:#116eae}table{border-collapse:collapse;display:block;overflow:auto;font-size:13px;margin:22px 0;width:100%}th,td{border:1px solid #dce4e9;padding:9px 11px;text-align:left;white-space:nowrap}th{background:#163f53;color:white}tr:nth-child(even){background:#f4f8fa}img{display:block;max-width:100%;height:auto;margin:24px auto}pre{background:#eef3f6;padding:20px;overflow:auto;line-height:1.65}code{font-size:.9em;overflow-wrap:anywhere}p{margin:16px 0}strong{color:#153f55}nav{position:sticky;top:0;background:#163f53;padding:9px 24px;color:white;z-index:2}nav a{color:white;margin-right:24px;font-size:14px}@media(max-width:700px){main{margin:0;padding:24px 18px}h1{font-size:25px}}@media print{body{background:white}main{margin:0;padding:0;box-shadow:none}nav{display:none}h2{break-before:page}table{font-size:10px}img{max-height:22cm;object-fit:contain}}
</style><nav><a href="#">CORE90 · 全面报告</a><a href="report.md">Markdown</a><a href="data/metrics.csv">32行总表</a><a href="data/grouped_metrics.csv">2784行分组数据</a><button onclick="window.print()">打印 / 保存PDF</button></nav><main>'''+body+'</main></html>'
(OUT/'report.html').write_text(doc,encoding='utf-8')
if Path(__file__).resolve()!=(OUT/'build_comprehensive_report.py').resolve():
    shutil.copy2(__file__,OUT/'build_comprehensive_report.py')
print(json.dumps({'report':str(OUT),'markdown_chars':len(text),'html_bytes':len(doc.encode()),'figures':8,'tables':text.count('|---'),'data_files':len(list((OUT/'data').iterdir()))},ensure_ascii=False))
