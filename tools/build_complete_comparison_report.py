"""Validate and aggregate every frozen score; no model selection or training access."""
from collections import defaultdict, Counter
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('E:/type10-7/local_artifacts/comparison_results_20260927')
CVS = Path('E:/type10-7/local_artifacts/cvs_matched_20260927')
RECOVERY = Path('E:/type10-7/local_artifacts/cvs_d92_recovery_20260928')
OUT = ROOT / 'docs/results/cvs_comparison_20260928'
SEEDS = {392005, 2026092701, 2026092702, 2026092703, 2026092704}
METHODS = ['cvcnn_ce-ce','cvcnn_ce-pl','riei_fd-ce','riei_fd-pl','drift-ce','drift-pl','poster-ce','radionet-ce','cvs-daot-rc4']
NAMES = dict(zip(METHODS,['CVCNN-CE','CVCNN-PL','RIEI-FD-CE','RIEI-FD-PL','DRIFT-CE','DRIFT-PL','POSTER','RadioNet-DF','CVS']))
MODES = {'support_ncm':'NCM','author_finetune':'FT','d92_registration':'D92','frozen_dg':'DG'}
ROUTES = [(m,'support_ncm') for m in METHODS[:-1]] + [('poster-ce','author_finetune'),('radionet-ce','author_finetune'),('cvs-daot-rc4','d92_registration')]
VIEWS = ['clean','practical_high','practical_mid','practical_low_urban']
METRICS = ['accuracy','macro_accuracy','old_accuracy','new_accuracy','harmonic_mean']


def load(path, count):
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['status'] == 'SCORED' and len(data['results']) == count, path
    return data['results']


def csvfile(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = load(BASE/'phase1_results.json',1280) + load(CVS/'phase1_final_results.json',160)
    p2 = load(BASE/'phase2_results.json',105840) + load(RECOVERY/'scored_results.json',10605)
    groups = defaultdict(lambda: defaultdict(list)); seen = set(); coverage = Counter(); split_sets = defaultdict(set)
    for r in p1:
        key = (r['row_id'],r['view'],r['receiver'])
        assert key not in seen; seen.add(key)
        method,seed = r['row_id'].rsplit('-s',1); seed=int(seed)
        cm=r['confusion']; assert sum(map(sum,cm))==r['count']
        assert abs(sum(cm[i][i] for i in range(6))/r['count']-r['accuracy'])<1e-12
        f1=mean(2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) if sum(cm[i])+sum(row[i] for row in cm) else 0 for i in range(6))
        assert abs(f1-r['macro_f1'])<1e-12
        for metric in ['accuracy','macro_f1']:
            groups[('phase1',method,'DG',r['view'],r['receiver'],0,0,metric)][seed].append(r[metric])
    seen=set()
    for r in p2:
        key=(r['row_id'],r['split_id'],r['mode'])
        assert key not in seen; seen.add(key)
        method,seed=r['row_id'].rsplit('-s',1);seed=int(seed)
        n=r['class_count']-6;k=0 if r['mode']=='frozen_dg' else r['k']
        assert n in [0,2,5,10,20] and r['query_count']==30*r['class_count']
        assert abs(r['accuracy']-r['macro_accuracy'])<1e-12
        if n:
            assert abs(r['accuracy']-(6*r['old_accuracy']+n*r['new_accuracy'])/(6+n))<1e-12
            h=2*r['old_accuracy']*r['new_accuracy']/(r['old_accuracy']+r['new_accuracy']) if r['old_accuracy']+r['new_accuracy'] else 0
            assert abs(h-r['harmonic_mean'])<1e-12
        else:
            assert r['new_accuracy'] is None and r['harmonic_mean'] is None
            assert r['accuracy']==r['old_accuracy']
        coverage[(method,r['mode'],seed)]+=1
        if r['mode']!='frozen_dg':split_sets[(method,r['mode'],seed)].add(r['split_id'])
        for scene in ['ALL',r['scenario']]:
            for rx in ['ALL',r['receiver']]:
                for metric in METRICS:
                    if r[metric] is not None:
                        groups[('phase2',method,r['mode'],scene,rx,n,k,metric)][seed].append(r[metric])
    for (method,mode,seed),count in coverage.items():
        assert count==(21 if mode=='frozen_dg' else 2100), (method,mode,seed,count)
    common=next(iter(split_sets.values()))
    assert len(common)==2100 and all(x==common for x in split_sets.values())
    summary={}; aggregated=[]; per_seed=[]
    for key,values in groups.items():
        stage,method,mode,scene,rx,n,k,metric=key
        assert set(values)==SEEDS, key
        expected=1 if stage=='phase1' else (3 if scene=='ALL' else 1)*(7 if rx=='ALL' else 1)*(1 if mode=='frozen_dg' else 5)
        assert all(len(v)==expected for v in values.values()),key
        byseed={s:mean(v)*100 for s,v in values.items()}
        fresh=[v for s,v in byseed.items() if s!=392005]
        meta=dict(stage=stage,method=method,mode=mode,scenario=scene,receiver=rx,new_classes=n,k=k,metric=metric)
        item=dict(**meta,fresh_mean=mean(fresh),fresh_sd=stdev(fresh),historical_392005=byseed[392005],records_per_model_seed=expected,fresh_model_seeds=4)
        summary[key]=item;aggregated.append(item)
        for seed,value in byseed.items():per_seed.append(dict(**meta,model_seed=seed,value_percent=value,records=expected))
    csvfile('aggregate_metrics.csv',aggregated);csvfile('per_model_seed_metrics.csv',per_seed)
    csvfile('phase1_all_records.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in p1])
    csvfile('phase2_all_records.csv',p2)
    validation=dict(status='PASS',phase1_records=len(p1),phase2_records=len(p2),common_registration_splits=len(common),duplicate_records=0,model_seeds=sorted(SEEDS),coverage=[dict(method=m,mode=mode,model_seed=s,records=c) for (m,mode,s),c in coverage.items()],checks=['all records parsed','unique row/split/mode','full five seed coverage','common 2100 split IDs across registration routes','expected records in every aggregation cell','query_count=30*class_count','overall/old/new weighted identity','harmonic mean identity','phase1 confusion/accuracy/Macro-F1 identity'])
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    def value(stage,m,mode,scene='ALL',n=0,k=0,metric='accuracy',rx='ALL',history=False):
        s=summary[(stage,m,mode,scene,rx,n,k,metric)]
        return f"{s['historical_392005']:.2f}" if history else f"{s['fresh_mean']:.2f} ± {s['fresh_sd']:.2f}"
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+['| '+' | '.join(row)+' |' for row in rows])
    text=['# CVS已执行批次对比实验结果（2026-09-28）',
        '**方法覆盖尚未齐全。**本报告完整汇总已执行批次，不代表论文所需的全部方法已经完成；原方案与补充方法见[方法总清单](../../CVS_COMPARISON_METHOD_CATALOG_20260930.md)。',
        '**完成状态：本批45个源模型、Phase1目标测试和Phase2固定矩阵均已完成。**D92恢复run全部5行完成2121条预测并独立评分；原两处FP32技术故障已修复，未删除失败记录、未重新选seed。',
        '## 1. 统计口径与版本',
        '所有表格单位为%，主表为4个固定新模型seed（2026092701至2026092704）的均值±样本标准差。历史优化seed392005单列。Phase2先在每个模型seed内平均7接收机×3场景×5次support抽样，再计算跨4个模型seed的统计，不能把105个episode当成105个独立模型seed。',
        'Phase1：原始DAOT A1＋FastTrust-RC4；所有方法使用第200轮最终权重。Phase2：CVS使用D92 E0去RF32，即P2-256-FULL、identity160＋FFT96。所有方法复用同一residual_noeq（post_sync残差，无额外均衡）received IQ和support/query划分。',
        'Phase1目标池为ManySig；Phase2为ManyTx。Phase1准确率不能直接减去Phase2准确率当成适应增益。合法适应增益应在Phase2同一仅旧类query池内，将适应后结果与该方法的frozen DG比较。',
        '## 2. 对比方法与适应路线',
        table(['组别','实现/用途','预算与权限'],[
            ['CVCNN、RIEI-FD、DRIFT的CE/PL','6条源训练路线；PL为无标签源数据伪标签扩展','Phase2统一冻结特征＋support NCM，不等同于各论文原生DA算法'],
            ['POSTER NCM、RadioNet-DF NCM','冻结骨干，归一化support类均值及余弦分类','仅合法support标签；query只前向'],
            ['POSTER FT','当前作者微调移植路线','Adam lr=0.001，10轮，batch=256，support-only'],
            ['RadioNet-DF FT','当前DF微调移植路线','Adam lr=0.001，30轮，batch=128，support-only；不代表ADA/triplet全部变体'],
            ['CVS-D92','原生support-only适应与联合注册','冻结特征＋合法source量化聚合ground；逐query面对全部注册类'],
            ['Frozen DG','每个源模型的零适应分类器','只用于6旧类；新类未注册时不能评估其识别能力']]),
        'CE/PL不是两种目标域适应算法，而是源训练监督方式的差异。NCM是本项目统一扩展；FT与D92分别列出，不把全部方法写成同一种微调。',
        '## 3. Phase1目标域结果']
    for metric,title in [('accuracy','准确率'),('macro_f1','Macro-F1')]:
        text += ['### '+title,table(['方法','clean','high','mid','low urban'],[[NAMES[m]]+[value('phase1',m,'DG',v,metric=metric) for v in VIEWS] for m in METHODS])]
    text += ['每seed：clean168000例、high55998例、mid56174例、low urban55828例。各星地场景覆盖互斥固定子集。',
        '## 4. Phase2仅旧类域适应（6旧类、0新类）',
        '这里的旧类准确率不含新类竞争；下节“注册后的旧类准确率”则面对旧＋新全部注册类别，不能混写。',
        table(['方法/路线','零适应DG','K=1','K=5','K=10','K=20'],[[NAMES[m]+'/'+MODES[mode],value('phase2',m,'frozen_dg')]+[value('phase2',m,mode,k=k,metric='old_accuracy') for k in [1,5,10,20]] for m,mode in ROUTES]),
        '### CVS按场景的旧类域适应',
        table(['场景','DG','K=1','K=5','K=10','K=20'],[[v,value('phase2','cvs-daot-rc4','frozen_dg',v)]+[value('phase2','cvs-daot-rc4','d92_registration',v,k=k,metric='old_accuracy') for k in [1,5,10,20]] for v in VIEWS[1:]]),
        '## 5. 新类注册：完整规模和K扫描',
        '每个episode含6个旧类及N个新类，每类K个support，query每类30例。每个query均面对6＋N个注册类。总体准确率A=(6×A_old+N×A_new)/(6+N)；H为旧/新准确率的调和均值。表中H先逐episode计算，再平均，不是把汇总后的旧/新均值再次代入。各类query数相同，因此本批macro_accuracy与总体准确率相同，但它不是Macro-F1。']
    for n in [2,5,10,20]:
        text += [f'### 6旧类＋{n}新类：总体准确率',table(['方法/路线','K=1','K=5','K=10','K=20'],[[NAMES[m]+'/'+MODES[mode]]+[value('phase2',m,mode,n=n,k=k) for k in [1,5,10,20]] for m,mode in ROUTES])]
    text += ['### CVS全部注册规模的分项指标',table(['新增类数','K','总体','旧类','新类','H'],[[str(n),str(k)]+[value('phase2','cvs-daot-rc4','d92_registration',n=n,k=k,metric=metric) for metric in ['accuracy','old_accuracy','new_accuracy','harmonic_mean']] for n in [2,5,10,20] for k in [1,5,10,20]]),
        '## 6. 各方法新类注册完整分项表']
    for n in [2,5,10,20]:
        for k in [1,5,10,20]:
            text += [f'### 6旧类＋{n}新类，K={k}',table(['方法/路线','总体','旧类','新类','H'],[[NAMES[m]+'/'+MODES[mode]]+[value('phase2',m,mode,n=n,k=k,metric=metric) for metric in ['accuracy','old_accuracy','new_accuracy','harmonic_mean']] for m,mode in ROUTES])]
    text += ['## 7. 结果解读与历史seed',
        'CVS仅旧类适应：零适应DG为69.62%，K=1为60.94%，下降8.68个百分点；K=5/10/20分别为72.36%/77.80%/80.92%。低仰角城市DG为52.82%，K=1为29.62%、K=5为49.26%，这两档仍低于DG，不能声称适应在所有场景均有效。这里只报告观察到的退化，未据query真值修改方法。',
        '6旧类＋20新类时，CVS K=5总体49.05%、新类47.21%、H=50.53%；K=20总体61.95%、新类60.19%、H=63.49%。K=5与K=20总体均值相对本表最佳外部对比分别高7.64和15.97个百分点。但K=1总体31.93%低于RIEI-FD-PL/NCM的32.66%，不能概括为全K最优。',
        '新增类别后旧类准确率下降同时包含注册类间竞争及适应过程影响，不能仅凭这一差值称为灾难性遗忘；冻结NCM同样可能因类别竞争下降。需要同一特征与分类器的分解对照才能作机制归因。',
        '### 历史seed392005单列',
        table(['方法','clean','high','mid','low urban'],[[NAMES[m]]+[value('phase1',m,'DG',v,history=True) for v in VIEWS] for m in METHODS]),
        '其余全部逐seed结果在per_model_seed_metrics.csv，包含历史seed的所有注册规模、K、场景、接收机和指标，不挑选最好seed。',
        '## 8. 完整性、缺项与完善建议',
        '本次全量解析Phase1的1440条评分记录与Phase2的116445条评分记录；对每条执行重复键、样本数、指标代数关系和模型seed覆盖校验。所有注册路线具有完全相同的2100个split ID。validation.json记录检查结果；该汇总验证不替代先前数据builder验证。',
        '目前完整的是已执行的固定矩阵。原方案的ProtoNet CDA、MRIOR-SDA、DADDA-SDA，以及CSIL、MoPC-HR、Orthogonal Incremental SEI尚无本批同协议结果；Feature Separation、Two-stage UDA、RadioNet ADA/Triplet及基础机制对照也须逐项记录。已找到部分方法的历史代码和实验定位，不能据旧检查表断言从未运行，也不能用历史不同数据、checkpoint或信道结果填入主表。完整范围、实现与版本差异见[方法总清单](../../CVS_COMPARISON_METHOD_CATALOG_20260930.md)。',
        '优先补强：①预登记等优化步数/等训练计算预算对照，披露现有CVS约222步/轮与外部基线约50步/轮的差异；②在同一CVS冻结特征上比较NCM、线性分类器和D92，分离表征收益与适应算法收益；③同一外部特征上补齐已预登记的原生DA方法，准确区分source-free、support-only与需要额外源数据/无标签目标数据的权限；④增加Phase2 Macro-F1、逐类召回与混淆矩阵，以及适应时延、峰值内存和部署字节数。以上尚无对应完整结果，不因本表目标成绩调参或选择性重跑。',
        'Phase2现有scorer未输出Macro-F1/逐类混淆矩阵，不能用macro_accuracy冒充。可对现有冻结预测做独立补充评分，无需重新训练；资源指标需要真实计时/内存记录，不能由准确率推算。4个新seed可提供描述性均值与标准差，但不足以直接宣称显著性或一区录用保证。',
        'POSTER FT等低分原样保留：当前结果仅对应本次移植实现、support规模和预算；不能据低分反向调整测试设置，也不能直接断言论文方法本身无效。',
        '## 9. 文件与复现入口',
        '- [aggregate_metrics.csv](aggregate_metrics.csv)：全部场景、接收机、N、K、指标的4新seed均值/样本标准差及历史seed值。\n- [per_model_seed_metrics.csv](per_model_seed_metrics.csv)：全部5模型seed逐项结果。\n- [phase1_all_records.csv](phase1_all_records.csv)、[phase2_all_records.csv](phase2_all_records.csv)：完整评分明细，可筛选而不需要再运行模型。\n- [validation.json](validation.json)：全量一致性检查。\n- 报告生成器：tools/build_complete_comparison_report.py。CSV为本地生成的数据产物，保留在原位置和工作区镜像，不进入Git；生成器、报告和检查摘要进入Git。',
        '来源：local_artifacts/comparison_results_20260927/phase1_results.json、phase2_results.json；local_artifacts/cvs_matched_20260927/phase1_final_results.json；local_artifacts/cvs_d92_recovery_20260928/scored_results.json。',
        'CVS Phase2恢复run：20260928-phase2-cvs-d92-practical-manytx-m5-r02；release commit c0e538d60c2e52a50112ee9b43337b4277d463d3。completion.json=SCORED、5×2121预测已独立读回；预测完成后才调用scorer，评分不回流训练或选模。']
    (OUT/'完整对比实验报告.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    concise={}
    for mode,n,k in [('frozen_dg',0,0),('d92_registration',0,1),('d92_registration',0,5),('d92_registration',0,10),('d92_registration',0,20),('d92_registration',20,5),('d92_registration',20,20)]:
        concise[f'{mode}/N{n}/K{k}']={metric:value('phase2','cvs-daot-rc4',mode,n=n,k=k,metric=metric) for metric in METRICS if ('phase2','cvs-daot-rc4',mode,'ALL','ALL',n,k,metric) in summary}
    print(json.dumps(dict(validation=validation['status'],cvs=concise,output=str(OUT)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
