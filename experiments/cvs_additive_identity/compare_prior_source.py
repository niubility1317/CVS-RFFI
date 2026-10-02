"""Paired E200 source comparison; no checkpoint, target or selection update."""
import argparse,csv,inspect,json,math,statistics
from pathlib import Path
from experiments.cvs_additive_identity.prepare import RUN,PROJECT
from experiments.cvs_additive_identity.publish import ssh
from experiments.cvs_coordinate_identity.model import coordinate_contract
from experiments.cvs_coordinate_identity.source import validate_config
from experiments.cvs_coordinate_identity.dispatch import read_source_record
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

PRIOR_RUN='20261002-phase1-cvs-coordinate-identity-manysig-m8-r01'
PRIOR_COMMIT='ecb268f7b4bcab111f041e177ecc3f967f1cb21a'
PAIRS={'additive_equivariant':'coordinate_equivariant','additive_gauge':'coordinate_gauge'}
SEEDS=(2026092701,2026092702,2026092703,2026092704)


def compare(root):
    e=root/'automation_reports/CV-SincNet'/RUN/'evidence'
    d=json.loads((e/'source_research_complete.json').read_text(encoding='utf-8'))
    from experiments.cvs_additive_identity.collect import validate_completed
    selection=validate_completed(d)
    # Re-read the actual old source metadata, not a target-bearing index or report.
    script='import json,math\nfrom pathlib import Path\nPROJECT='+repr(PROJECT)+'\n'
    script+='contracts='+repr({v:coordinate_contract(v) for v in PAIRS.values()})+'\n'
    script+='def coordinate_contract(v):return contracts[v]\n'
    script+=inspect.getsource(validate_config)+'\n'+inspect.getsource(read_source_record)+'\n'
    script+='original=json.loads(Path('+repr(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')+').read_text())\n'
    rows=[dict(variant=v,model_seed=s,source_output=PROJECT+'/runs/'+PRIOR_RUN+'/'+v+'-s'+str(s)+'/source') for v in PAIRS.values() for s in SEEDS]
    script+='rows='+repr(rows)+'\nrecords=[]\nVARIANTS=tuple(contracts)\n'
    script+='for row in rows:\n q=Path(row["source_output"]);resolved=json.loads((q/"resolved_config.json").read_text())\n'
    script+=' if resolved["commit"]!='+repr(PRIOR_COMMIT)+' or resolved["backend_flags"]!='+repr(FULL_FP32_POLICY)+':raise ValueError("Prior source actual commit/precision mismatch")\n'
    script+=' records.append(read_source_record(row,original,"cvs_coordinate_identity"))\nprint(json.dumps(records))\n'
    old=json.loads(ssh(script))
    if len(old)!=8 or {(r['variant'],r['seed']) for r in old}!={(v,s) for v in PAIRS.values() for s in SEEDS}:raise ValueError('Incomplete prior eight source records')
    comparison=[]
    for r in d['rows']:
        current=r['source_record'];v=current['variant'];s=current['seed'];prior=next(a for a in old if a['variant']==PAIRS[v] and a['seed']==s)
        if current['parameters']!=prior['parameters']:raise ValueError('Claimed equal-parameter comparison differs')
        comparison.append(dict(variant=v,prior_variant=PAIRS[v],model_seed=s,parameters=current['parameters'],
            prior_V=prior['accuracy'],V=current['accuracy'],V_delta_pp=100*(current['accuracy']-prior['accuracy']),
            prior_worst_RX=prior['worst_rx'],worst_RX=current['worst_rx'],worst_RX_delta_pp=100*(current['worst_rx']-prior['worst_rx'])))
    summary=[]
    for v in PAIRS:
        rows=[r for r in comparison if r['variant']==v]
        summary.append(dict(variant=v,prior_variant=PAIRS[v],V_delta_pp_mean=statistics.mean(r['V_delta_pp'] for r in rows),
            V_delta_pp_seed_sd=statistics.stdev(r['V_delta_pp'] for r in rows),worst_RX_delta_pp_mean=statistics.mean(r['worst_RX_delta_pp'] for r in rows),
            worst_RX_delta_pp_seed_sd=statistics.stdev(r['worst_RX_delta_pp'] for r in rows)))
    artifact=dict(status='VERIFIED',source_only=True,used_for_selection=False,target_access=False,checkpoint_loaded=False,
        prior_run=PRIOR_RUN,prior_commit=PRIOR_COMMIT,new_run=RUN,prior_actual_records=old,rows=comparison,summary=summary,
        frozen_selection_unchanged=selection,interpretation='Same physical source roles, E200, model seeds, fullFP32 and parameter counts;paired source diagnostic,not new blind RX test or proof of identified TX causality')
    (e/'paired_previous_source.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (e/'paired_previous_source.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    p=e.parent/'report.md';text=p.read_text(encoding='utf-8')
    text+='\n## 与上一版乘性坐标模型的源域配对\n\n实际重读上一版8份源E200 metadata，原物理角色、预算、scratch、完整FP32及commit核实一致；同核心、同model seed、同参数量对比，只用于源机制诊断，不增加本轮候选或改变已经冻结的源选择。\n\n| 核心 | Δ源V均值±seedSD（百分点） | Δ最差源RX均值±seedSD（百分点） |\n|---|---:|---:|\n'
    for r in summary:text+=f"| {r['variant']} | {r['V_delta_pp_mean']:+.4f}±{r['V_delta_pp_seed_sd']:.4f} | {r['worst_RX_delta_pp_mean']:+.4f}±{r['worst_RX_delta_pp_seed_sd']:.4f} |\n"
    text+='\nΔ为本轮加性减上一版乘性，源RX不是未知RX测试；这项比较不能证明TX/RX硬件因果分离。[8行源配对](evidence/paired_previous_source.csv) · [实际来源核实](evidence/paired_previous_source.json)。\n'
    p.write_text(text,encoding='utf-8');print(json.dumps(dict(status='VERIFIED',paired_rows=8,summary=summary,target_access=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);compare(p.parse_args().root)
