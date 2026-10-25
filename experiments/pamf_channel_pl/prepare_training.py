"""Prepare source, prediction and truth-side configs without sharing truth upstream."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.practical_full_2x2.plan import TASKS
from .training_plan import TRAIN_RUN,rows,make_config
from .plan import validate_context


def prepare(rt):
    context=read(rt['context']);by=validate_context(context);reference=context['reference']
    proof=read(Path(context['source_diagnostic_root'])/'complete.json')
    if (proof['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or proof['total']!=33 or proof['passed']!=33
            or proof['target_reads'] or proof['model_updates'] or proof['U_truth_read']):
        raise ValueError('Fixed scientific source falsification did not pass for the whole candidate')
    tested={(r['protocol'],r['model_seed']):r for r in proof['rows']}
    root=Path(rt['run_root']);jobs=[];sources={};refs={}
    def add(kind,row,c,completion,status,deps=()):
        rid=row['row_id'];path=root/'configs'/(kind+'__'+rid+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+rid,kind=kind,row_id=rid,model_seed=row['model_seed'],gpu=row['gpu'],
            config=str(path),completion=str(completion),successful_statuses=[status],dependencies=list(deps)))
    for row in rows():
        p=row['protocol'];s=row['model_seed'];rid=row['row_id'];case=by[p,s];h=reference['handles'][p]
        evidence=tested[p,s]
        if (evidence['checkpoint']!=case['parent_source']+'/final_ssdg.pth'
                or evidence['source_V_diagnostic_reads']!=read(h['source_contract'])['counts']['V']):
            raise ValueError('Source evidence belongs to another parent/role')
        c=make_config(case,rt['project'],rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        if read(c['source_contract'])!=read(h['source_contract']):raise ValueError('Source/evaluation contract differs')
        add('train',row,c,Path(c['output_root'])/'complete.json','SOURCE_FROZEN')
        sources[rid]=c['output_root'];out=root/'readouts'/rid
        pred=dict(row_id=rid,model_seed=s,protocol_id=p,run_id=TRAIN_RUN,method='ChannelPL',
            source_output=c['output_root'],source_contract=h['source_contract'],classes=h['classes'],source_days=[1,2,3],
            tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
            restored_caches=dict(LU=reference['caches'][p]['LU']['path']),readouts=['raw','DURA-LU'],
            output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
        add('predict',row,pred,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[str(Path(c['output_root'])/'complete.json')])
        scorer=deepcopy(pred);scorer.update(prediction_output=pred['output_root'],output_root=str(out/'scores'),tests=h['tests'])
        for key in ('source_output','source_contract','restored_caches','frozen_sources'):scorer.pop(key)
        add('score',row,scorer,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
        refs[rid]={method:context['reference_run_root']+'/readouts/'+p+'-'+method+'-s'+str(s)+'/scores/scores.json'
            for method in ('supervised','ordinary_PL','REPL')}
    write(root/'jobs.json',jobs);write(root/'reference_scores.json',refs);write(root/'sources.json',sources)
    write(root/'preparation_complete.json',dict(status='PREPARED',new_F=33,reused_E20=33,
        new_training_epochs=990,prediction_jobs=33,score_jobs=33,new_G=0,new_IQ_cache=0,
        populations=list(TASKS),readouts=['raw','DURA-LU'],no_target_feedback=True))
    return jobs
