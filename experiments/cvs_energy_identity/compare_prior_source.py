"""Same-core source comparison changing only normalization axes; never changes selection."""
import argparse,csv,inspect,json,statistics
from pathlib import Path
from experiments.cvs_energy_identity.prepare import RUN,PROJECT
from experiments.cvs_energy_identity.publish import ssh
from experiments.cvs_energy_identity.dispatch import read_source_record,SEEDS
from experiments.cvs_energy_identity.model import VARIANTS
from experiments.cvs_fractional_identity.model import fractional_contract
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

PRIOR_RUN='20261002-phase1-cvs-fractional-identity-manysig-m8-r01'
PRIOR_VARIANT='fractional_half'

def paired_summary(new_rows,prior):
    if len(prior)!=4 or {(r['variant'],r['seed']) for r in prior}!={(PRIOR_VARIANT,s) for s in SEEDS}:
        raise ValueError('Incomplete or duplicated fixed-half source controls')
    selected=[r for r in new_rows if r['source_record']['variant']=='energy_half']
    if len(new_rows)!=8 or {(r['source_record']['variant'],r['source_record']['seed']) for r in new_rows}!={(v,s) for v in VARIANTS for s in SEEDS}:
        raise ValueError('Incomplete energy source matrix')
    paired=[]
    for row in selected:
        c=row['source_record'];p=next(x for x in prior if x['seed']==c['seed'])
        if p['parameters']!=202553 or c['parameters']!=202553 or row['epochs'][-1]['alignment_strength']!=.5:
            raise ValueError('Matched core/parameter/fixed-alpha claim differs')
        paired.append(dict(variant=c['variant'],prior_variant=PRIOR_VARIANT,model_seed=c['seed'],parameters=c['parameters'],prior_parameters=p['parameters'],alignment_strength=.5,prior_alignment_strength=.5,V=c['accuracy'],prior_V=p['accuracy'],V_delta_pp=100*(c['accuracy']-p['accuracy']),worst_RX=c['worst_rx'],prior_worst_RX=p['worst_rx'],worst_RX_delta_pp=100*(c['worst_rx']-p['worst_rx'])))
    summary=[dict(variant='energy_half',prior_variant=PRIOR_VARIANT,V_delta_pp_mean=statistics.mean(r['V_delta_pp'] for r in paired),V_delta_pp_seed_sd=statistics.stdev(r['V_delta_pp'] for r in paired),worst_RX_delta_pp_mean=statistics.mean(r['worst_RX_delta_pp'] for r in paired),worst_RX_delta_pp_seed_sd=statistics.stdev(r['worst_RX_delta_pp'] for r in paired),alignment_strength_mean=.5)]
    return paired,summary


def compare(root):
    e=root/'automation_reports/CV-SincNet'/RUN/'evidence';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    d=read(e/'source_research_complete.json')
    from experiments.cvs_energy_identity.collect import validate_completed
    selection=validate_completed(d)
    # Only source-only historical artifacts. No report/index/test/checkpoint reads.
    rows=[dict(variant=PRIOR_VARIANT,model_seed=s,source_output=PROJECT+'/runs/'+PRIOR_RUN+'/'+PRIOR_VARIANT+'-s'+str(s)+'/source') for s in sorted(SEEDS)]
    expected=fractional_contract(PRIOR_VARIANT)
    script='import json,math\nfrom pathlib import Path\nPROJECT='+repr(PROJECT)+'\n'+inspect.getsource(read_source_record)+'\n'
    script+='original=json.loads(Path('+repr(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')+').read_text())\nrows='+repr(rows)+'\nrecords=[]\n'
    script+='for row in rows:\n q=Path(row["source_output"]);resolved=json.loads((q/"resolved_config.json").read_text());done=json.loads((q/"completion.json").read_text())\n'
    script+=' if resolved["backend_flags"]!='+repr(FULL_FP32_POLICY)+' or done["backend_flags"]!='+repr(FULL_FP32_POLICY)+':raise ValueError("Prior actual source precision mismatch")\n'
    script+=' if resolved.get("fractional_actual")!='+repr(expected)+' or resolved.get("fractional")!='+repr(expected)+' or resolved.get("fractional_active") is not True:raise ValueError("Prior fixed-half physical operator mismatch")\n'
    script+=' record=read_source_record(row,original,"cvs_fractional_identity");record["actual_commit"]=resolved["commit"];records.append(record)\nprint(json.dumps(records))\n'
    prior=json.loads(ssh(script));paired,summary=paired_summary(d['rows'],prior)
    actual_prior_commits={r['actual_commit'] for r in prior}
    if len(actual_prior_commits)!=1 or not all(len(c)==40 and set(c)<=set('0123456789abcdef') for c in actual_prior_commits):raise ValueError('Unverified historical immutable source commit')
    historical=read(root/'automation_reports/CV-SincNet'/PRIOR_RUN/'evidence/source_research_complete.json')
    old_rows=[r for r in historical['rows'] if r['resolved']['variant']==PRIOR_VARIANT]
    if len(old_rows)!=4:raise ValueError('Missing previously audited same-core source evidence')
    for record in prior:
        old=next(r for r in old_rows if r['resolved']['model_seed']==record['seed'])
        if (record['actual_commit']!=old['resolved']['commit'] or record['accuracy']!=old['completion']['final_source_metrics']['source_val_accuracy'] or record['worst_rx']!=old['completion']['final_source_metrics']['source_val_worst_rx'] or record['parameters']!=old['profile']['total_parameters']):
            raise ValueError('Actual historical source metadata changed from full audited evidence')
    geometry=[]
    for new in [r for r in d['rows'] if r['resolved']['variant']=='energy_half']:
        seed=new['resolved']['model_seed'];old=next(r for r in old_rows if r['resolved']['model_seed']==seed)
        rec=dict(model_seed=seed)
        for prefix,row in [('shared',new),('per_channel',old)]:
            diag=row['source_diagnostics'];physical=row['physical_diagnostics']
            between=diag['mean_between_tx_centroid_squared_distance'];within=diag['mean_within_tx_rx_centroid_squared_distance']
            rec[prefix+'_source_between_TX_squared_distance']=between
            rec[prefix+'_source_within_TX_cross_RX_squared_distance']=within
            rec[prefix+'_source_TX_to_RX_geometry_ratio']=between/max(within,1e-12)
            tx_changes=physical['isolated_tx_changes'];rx_changes=[a for a in physical['records'] if a['rx']!='identity']
            if len(tx_changes)!=4 or len(rx_changes)!=25:raise ValueError('Incomplete identical public TX/RX diagnostic')
            rec[prefix+'_public_isolated_TX_embedding_distance_mean']=statistics.mean(a['unit_embedding_distance_to_ideal_tx'] for a in tx_changes)
            rec[prefix+'_public_same_TX_cross_RX_embedding_distance_mean']=statistics.mean(a['unit_embedding_distance_same_tx_identity_rx'] for a in rx_changes)
        geometry.append(rec)
    artifact=dict(status='VERIFIED',source_only=True,used_for_selection=False,target_access=False,checkpoint_loaded=False,prior_run=PRIOR_RUN,prior_actual_records=prior,new_run=RUN,rows=paired,summary=summary,source_and_public_geometry=geometry,frozen_selection_unchanged=selection,interpretation='Same core, fixedalpha0.5, physical source roles, model/loader seed, E200x50, CE/noaugmentation, fullFP32 and 202553 parameters; only per-channel versus shared denominator changes. Separate historical training with nondeterministic kernels and different GPU assignments; not TX/RX causal separation or blind RX test. Geometry is descriptive only; source labels are actual TX identities, public TX settings are not identity labels; TX/RX distance ratio is not causal identification.')
    (e/'paired_normalization_source.json').write_text(json.dumps(artifact,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    with (e/'paired_normalization_source.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    p=e.parent/'report.md';text=p.read_text(encoding='utf-8')
    text+='\n## 同核心半校正的归一化源域配对\n\n实际重读此前4份fractional_half源E200记录，完整物理角色、预算、scratch、model/loader seed、完整FP32和半校正前端匹配。同一equivariant_memory核心比较；两者固定alpha0.5、202553参数；结构差异仅归一化轴，历史独立训练/不同GPU/非确定核仍可能有数值差异。raw候选的历史核心未登记同一完整FP32，因此不声明raw的单因素配对。独立历史训练，不加载旧权重，不改变本轮冻结12记录候选排序。\n\n| 候选 | 最终alpha均值 | Δ源V均值±seedSD（百分点） | Δ最差源RX均值±seedSD（百分点） |\n|---|---:|---:|---:|\n'
    for r in summary:text+=f"| {r['variant']} | {r['alignment_strength_mean']:.8g} | {r['V_delta_pp_mean']:+.4f}±{r['V_delta_pp_seed_sd']:.4f} | {r['worst_RX_delta_pp_mean']:+.4f}±{r['worst_RX_delta_pp_seed_sd']:.4f} |\n"
    text+='\n差分为共享分母减逐通道分母（两者alpha=0.5），只说明源域关联，不是独有TX频偏或硬件因果分离证明。[4行配对](evidence/paired_normalization_source.csv) · [实际源来源](evidence/paired_normalization_source.json)。\n'
    text+='\n|seed|源TX/RX几何比：逐通道→共享|公共孤立TX距离：逐通道→共享|公共同TX跨RX距离：逐通道→共享|\n|---|---:|---:|---:|\n'
    for a in geometry:text+=f"|{a['model_seed']}|{a['per_channel_source_TX_to_RX_geometry_ratio']:.6g}→{a['shared_source_TX_to_RX_geometry_ratio']:.6g}|{a['per_channel_public_isolated_TX_embedding_distance_mean']:.6g}→{a['shared_public_isolated_TX_embedding_distance_mean']:.6g}|{a['per_channel_public_same_TX_cross_RX_embedding_distance_mean']:.6g}→{a['shared_public_same_TX_cross_RX_embedding_distance_mean']:.6g}|\n"
    text+='\n源几何比为真实TX类中心间平方距离/同TX跨RX中心偏移。公共TX距离使用四个孤立硬件设置变化，RX距离使用25个同TX/非identityRX组合；公共设置不是实际TX标签。距离只描述敏感性和源关联，不进入源选择，也不证明真实硬件因果分离。\n'
    p.write_text(text,encoding='utf-8');print(json.dumps(dict(status='VERIFIED',paired_rows=4,summary=summary,target_access=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);compare(p.parse_args().root)
