"""All three clean cells fixed before the separate truth-side process starts."""
from pathlib import Path
from . import design as d

def physical_ids():
    import numpy as np
    manifest=d.read(Path(d.CAPSULE)/'manifest.json')
    expected=dict(status='VALIDATED_ONCE',count=d.COUNT,splits=d.SPLITS,classes=d.CLASSES,source_query_disjoint=True,unique_ids=True,views=['clean'])
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Historical capsule contract differs')
    with np.load(Path(d.CAPSULE)/'index.npz',allow_pickle=False) as v:ids=v['ids'].copy()
    if ids.shape!=(d.COUNT,) or len(set(ids))!=d.COUNT:raise ValueError('Historical opaque IDs differ')
    return ids

def configure():
    from experiments.cvs_ce_epoch2040 import evaluate as e
    e.d=d;e.COUNT=d.COUNT;e.shared.d=d;e.shared.COUNT=d.COUNT;e.shared.physical_ids=physical_ids
    return e

def frozen_source():
    return configure().snapshot(d.rows()[0]['row_id'])[0]

def predict():
    configure().predict(d.rows()[0]['row_id'])

def preflight():
    e=configure();configs=e.snapshot(d.rows()[0]['row_id'])
    return e.prediction_preflight(configs)

def metric(y,pred,classes=6):
    import numpy as np
    cm=np.zeros((classes,classes),dtype=np.int64)
    np.add.at(cm,(y,pred),1)
    diagonal=cm.diagonal();denom=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*diagonal,denom,out=np.zeros(classes,dtype=float),where=denom>0)
    return dict(count=len(y),accuracy=float((y==pred).mean()),Macro_F1=float(f1.mean()),confusion=cm.tolist())

def score():
    import numpy as np
    ids=preflight()  # Complete predictions and source provenance before truth.
    out=d.BASE/'scores'
    if out.exists():raise FileExistsError('Existing scoring; reconcile')
    with np.load(d.BASE/d.rows()[0]['row_id']/'prediction/predictions.npz',allow_pickle=False) as v:
        pred=v['clean'].copy();logp=v['clean_log_probabilities'].copy()
    with np.load(d.TRUTH,allow_pickle=False) as v:
        if not np.array_equal(v['ids'],ids):raise ValueError('Truth join differs')
        y=v['y'].copy();rx=v['rx'].copy();days=v['days'].copy();splits=v['splits'].copy()
    if y.shape!=(d.COUNT,) or y.min()!=0 or y.max()!=5 or {s:int((splits==s).sum()) for s in d.SPLITS}!=d.SPLITS:raise ValueError('Historical truth dimensions differ')
    records=[]
    for split,count in d.SPLITS.items():
        whole=splits==split
        for dimension,values in [('overall',[None]),('receiver',sorted(set(rx[whole].tolist()))),('day',sorted(set(days[whole].tolist()))),('TX',range(6))]:
            for value in values:
                mask=whole if value is None else whole&(rx==value if dimension=='receiver' else days==value if dimension=='day' else y==value)
                result=metric(y[mask],pred[mask]);result.update(split=split,dimension=dimension,stratum=d.CLASSES[value] if dimension=='TX' else value,
                    NLL=float(-logp[np.arange(d.COUNT)[mask],y[mask]].mean()),model_seed=392002,epochs=20)
                records.append(result)
    d.write(out/'metrics.json',dict(status='SCORED_COMPLETE',records=records,truth_last=True))
    source=d.read(d.BASE/d.rows()[0]['row_id']/'source/completion.json')
    d.write(out/'resources.json',dict(training_seconds=source['elapsed_seconds'],optimizer_updates=4440,epochs=20,prediction=d.read(d.BASE/d.rows()[0]['row_id']/'prediction/complete.json')))
    overall=[r for r in records if r['dimension']=='overall']
    lines=['# E20 旧门控＋cos：历史 ADV3B02 划分','', '状态：训练和三个 clean 测试集合均已完成，prediction 固定后独立 truth-last 评分。','',
        'seed=392002；从零训练 E20/4440 更新；纯 CE、label_smoothing=0；无 U/EMA/星地训练增强。',
        '历史 L/U/V=8400/58800/16800，标注率为训练池的12.5%；这是用户指定的历史比较，不冒充当前10%标注协议。',
        '与历史 ADV3B02 E200 比较时，差异同时包含网络、损失和训练预算，不把全部收益归因于网络结构。','',
        '|测试集合|样本数|准确率 (%)|Macro-F1 (%)|','|---|---:|---:|---:|']
    lines += [f"|{r['split']}|{r['count']}|{100*r['accuracy']:.4f}|{100*r['Macro_F1']:.4f}|" for r in overall]
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'scoring_complete.json',dict(status='SCORED_COMPLETE',models=1,views=['clean'],splits=d.SPLITS,prediction_count=d.COUNT,truth_last=True,independent_recount='VERIFIED',overall=overall))
    print('HISTORICAL CLEAN SCORED '+str([(r['split'],r['accuracy']) for r in overall]),flush=True)
