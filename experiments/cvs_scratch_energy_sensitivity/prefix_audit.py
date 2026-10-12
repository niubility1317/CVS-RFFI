"""Read every complete record in one live source-log snapshot, without mutations."""
import argparse,json,math,time,zlib
from pathlib import Path
from . import design as d

REMOTE=r'''
import sys,json,time,zlib,math
from pathlib import Path
sys.path.insert(0,RELEASE)
from experiments.cvs_scratch_energy_sensitivity import design as d
from experiments.cvs_scratch_energy_sensitivity.analyze_source import _style_search,_check_calibration,Moments
from experiments.cvs_receiver_residual_capacity4.control import process
def read(p):return json.loads(p.read_text()) if p.is_file() else None
launch=read(d.BASE/'launch.json');registered={r['row_id']:r for r in launch['rows']};rows=[]
for row in d.rows():
 c=d.config(row);source=Path(c['output_root']);path=source/'step_metrics.jsonl.gz'
 if not path.is_file():continue
 compressed=path.read_bytes();decoder=zlib.decompressobj(31);raw=decoder.decompress(compressed);last=raw.rfind(b'\n')
 lines=raw[:last+1].splitlines() if last>=0 else [];aux=0;styles=0;identity=0;stats={};group_stats={};layer_stats={};first=None;last_aux=None
 cal=_check_calibration(source,c) if (source/'calibration.json').is_file() else None
 def add(bucket,key,value):bucket.setdefault(key,Moments()).add(value)
 for n,line in enumerate(lines,1):
  item=json.loads(line);assert item['step']==n;metrics=item['metrics'];detail=item['details'];active=n>444
  assert all(math.isfinite(v) for v in metrics.values() if isinstance(v,(int,float)))
  assert bool(detail['mixstyle_steps'])==active and detail['teacher_forwards']==0
  if active:
   style=_style_search(detail['L_style'],c,n,detail['ramp']);styles+=1;identity+=int(style['identity_selected'])
   for k in ('weighted_ce','weighted_KL','weighted_style_ce','weighted_fishr','total_loss'):add(stats,k,metrics[k])
  expected=active and n%4==0;assert bool(detail['auxiliary_steps'])==expected
  if expected:
   assert cal is not None;aux+=1;first=first or n;last_aux=n
   multi=detail['multilevel'];assert multi['fishr_levels']==['time_down','t2.pool']
   assert multi['fixed_scales']==cal['fixed_scales'] and multi['sensitivity_objective']==c['multilevel']['sensitivity_objective']
   assert multi['sensitivity_normalization']=='fixed_layer_source_RMS' and multi['aux_forwards']==2
   cap=detail['joint_gradient_cap'];assert cap['applied'] and cap['scalar_detached'] and cap['complete_disjoint_parameter_coverage']
   assert cap['parameter_tensors']==67 and 0<=cap['scalar']<=1 and cap['rho']==.2
   add(stats,'cap_scalar',cap['scalar']);add(stats,'cap_limited',int(cap['scalar']<1));add(stats,'all_R_groups_zero',int(cap['all_regularizer_groups_zero']))
   for group,values in cap['groups'].items():
    b=group_stats.setdefault(group,{})
    assert values['capped_R']<=values['norm_bound']*(1+1e-6)+1e-12
    assert math.isclose(values['capped_R'],cap['scalar']*values['raw_R'],rel_tol=1e-7,abs_tol=1e-12)
    for key in ('norm_id','raw_R','capped_R'):add(b,key,values[key])
    add(b,'capped_R_over_ID',values['capped_R']/values['norm_id'] if values['norm_id']>0 else None)
    add(b,'limiting',int(group in cap['active_bound_groups']))
   for layer,info in multi['levels'].items():
    b=layer_stats.setdefault(layer,{})
    add(b,'raw_fishr',info['raw_fishr'])
    for split in ('A','B'):
     sense=info['sensitivity_'+split]
     assert sense['fixed_scale']==cal['fixed_scales'][layer] and sense['score_requires_grad']
     assert len(sense['raw_RMS_by_RX_TX'])==30 and all(v['n']==2 for v in sense['raw_RMS_by_RX_TX'].values())
     for key in ('raw_sensitivity_rms_mean','raw_sensitivity_rms_min','raw_sensitivity_rms_max','normalized_score_rms_mean','low_norm_rate'):add(b,split+'/'+key,sense[key])
 assert aux==max(0,len(lines)//4-111) and styles==max(0,len(lines)-444)
 def finish(b):return {k:v.result() for k,v in b.items()}
 receipt=registered[row['row_id']]
 rows.append(dict(row_id=row['row_id'],pid=receipt['pid'],process=process(receipt['pid']),source_prefix_steps=len(lines),active_steps=styles,identity_replays=identity,style_replays=styles-identity,
  auxiliary_steps=aux,first_aux_step=first,last_aux_step=last_aux,calibration_status=cal['status'] if cal else None,fixed_scales=cal['fixed_scales'] if cal else None,
  statistics=finish(stats),groups={k:finish(v) for k,v in group_stats.items()},layers={k:finish(v) for k,v in layer_stats.items()},gzip_finished=decoder.eof,captured_bytes=len(compressed)))
print(json.dumps(dict(status='VERIFIED_SOURCE_PREFIX',run_id=d.RUN,read_at=time.time(),source_only=True,target_access=False,
 scope='Every complete JSON line in each captured live source prefix; no final-budget or performance claim',rows=rows,total_steps=sum(r['source_prefix_steps'] for r in rows))))
'''

def capture(output):
    from experiments.cvs_phase1_stack.publish import ssh
    output=Path(output)
    if output.exists():raise FileExistsError('Existing prefix retained; choose a new evidence path')
    payload=REMOTE.replace('RELEASE',repr(d.PROJECT+'/releases/'+d.RELEASE))
    wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True)\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
    value=json.loads(ssh(wrapper));d.write(output,value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=capture(a.output)
    print(json.dumps(dict(status=v['status'],total_steps=v['total_steps'],rows=[dict(row=r['row_id'],steps=r['source_prefix_steps'],aux=r['auxiliary_steps'],first=r['first_aux_step'],cap_scalar_mean=r['statistics'].get('cap_scalar',{}).get('mean'),group_R_ID_mean={k:x['capped_R_over_ID']['mean'] for k,x in r['groups'].items()}) for r in v['rows']]),ensure_ascii=False))
