"""Independent post-score R56 arithmetic; cooperative shared CPU score lock.

Read immutable features and summaries only after full completion. No neural
prediction, checkpoint, F access, fitting, or model selection is performed.
"""

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--release',required=True)
    parser.add_argument('--controller-pid',type=int,required=True)
    args=parser.parse_args()
    RUNROOT,RELEASE,PID=args.run_root,args.release,args.controller_pid
    import json,os,sys,time
    from pathlib import Path
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    import numpy as np
    root=Path(RUNROOT);base=root.parent.parent
    sys.path.insert(0,RELEASE)
    from experiments.pamf_compression_formal_identity import design as d
    import fcntl
    lock=(base/'runs'/d.SCORING_RUN/'score.lock').open('a+b')
    fcntl.flock(lock,fcntl.LOCK_EX)
    def read(path):return json.loads(path.read_text())
    def state(pid):
     p=Path('/proc')/str(pid)
     try:
      text=(p/'stat').read_text()
      if text[text.rfind(')')+2:].split()[0]=='Z':return 'EXITED_ZOMBIE'
      argv=(p/'cmdline').read_bytes().decode().split(chr(0))
      return 'LIVE' if 'experiments.pamf_compression_formal_identity.runner' in argv and root.name in ' '.join(argv) else 'EXITED_PID_REUSED'
     except FileNotFoundError:return 'EXITED'
    done=read(root/'complete.json');assert done['status']=='ANALYZED' and not (root/'failure.json').exists()
    launch=read(root/'launch.json');launches={k:read(root/(k+'_launch.json')) for k in ('extract','score')}
    processes=dict(controller=state(PID),**{k:state(v['pid']) for k,v in launches.items()});assert all(v.startswith('EXITED') for v in processes.values())
    smoke=read(root/'smoke.json');assert smoke['status']=='PASS' and smoke['model_checkpoint_loads']==smoke['F_access']==smoke['TX_labels_read']==smoke['target_reads']==0
    s=read(root/'summary.json');config=read(root/'resolved_config.json');meta=read(root/'feature_complete.json');cal=read(root/'calibration_reference.json')
    assert config==s['configuration'] and config['commit']==launch['commit']
    assert s['all_features_before_labels'] and s['target_truth_for_evaluation_only'] and s['target_truth_reads']==1
    assert not s['parent_training_or_decisions_changed'] and not s['source_selection_changed'] and not s['new_source_thresholds'] and s['selected_candidate'] is None
    for key in ('F_access','F_weights_read','F_embeddings_logits_read','model_checkpoint_loads','extra_model_predictions','neural_training_updates','head_fit_updates','new_calibration_samples'):assert s[key]==0
    assert cal['count']==cal['L_count']+cal['U_count']==8192 and cal['new_calibration_samples']==cal['new_parameters_fitted']==0
    assert meta['cells']==280 and meta['TX_labels_read']==meta['F_access']==meta['model_checkpoint_loads']==0
    views=tuple(s['views']);sizes=s['groups'];candidates=tuple(s['candidates']);readouts=('raw',*candidates)
    assert len(candidates)==9
    assert meta['reused_feature_cells']==84 and meta['new_feature_cells']==196 and meta['analytic_transform_samples']==8232000
    ids=np.load(root/'physical_ids.npy',allow_pickle=False);assert ids.shape==(168000,) and len(np.unique(ids))==168000
    keys={(v,r,g) for v in views for r in readouts for g in sizes}
    assert {p.stem for p in (root/'features').glob('*.npy')}=={'__'.join(k) for k in keys}
    features={k:np.load(root/'features'/('__'.join(k)+'.npy'),mmap_mode='r',allow_pickle=False) for k in keys}
    for (v,r,g),a in features.items():assert a.shape==(168000,sizes[g]) and a.dtype==np.float64 and np.isfinite(a).all()
    from experiments.pamf_compression_formal_identity import design as d
    assert candidates==d.CANDIDATES
    formal=base/'runs'/d.FORMAL/'reference'
    assert np.array_equal(ids,np.load(formal/'physical_ids.npy',allow_pickle=False))
    assert all(read(base/'runs'/run/'complete.json')['status']=='ANALYZED' for run in d.PARENTS)
    cached=base/'runs'/d.CACHED_RUN
    assert read(cached/'launch.json')['commit']==d.CACHED_COMMIT and read(cached/'complete.json')['status']=='ANALYZED'
    assert np.array_equal(ids,np.load(cached/'physical_ids.npy',allow_pickle=False))
    for v in views:
     for name in readouts:
      for group in sizes:
       path=root/'features'/(v+'__'+name+'__'+group+'.npy')
       if name in d.CACHED_READOUTS:
        assert path.is_symlink() and path.resolve()==(cached/'features'/(v+'__'+d.CACHED_READOUTS[name]+'__'+group+'.npy')).resolve()
       else:assert not path.is_symlink()

    # Truth read only after all three parents full formal scores and all280 fixed finite matched features.
    truth=read(base/'runs'/d.TRUTH_RUN/'data/truth.json')
    y=np.asarray([truth[str(pid)]['label'] for pid in ids],dtype=np.int64)
    rx=np.asarray([str(truth[str(pid)]['receiver']) for pid in ids]);day=np.asarray([str(truth[str(pid)]['day']) for pid in ids]);n=len(ids)
    assert len(set(rx))==7 and len(set(day))==4 and set(y)==set(range(6))
    groups={('overall','all'):np.ones(n,bool)}
    for dim,value in [('TX',y),('RX',rx),('day',day)]:groups.update({(dim,str(v)):value==v for v in np.unique(value)})
    assert len(s['records'])==len(groups)*len(views)*len(sizes)*len(candidates)==4536
    rng=np.random.default_rng(2026102034);permuted=np.tile(y,(199,1))
    for r in np.unique(rx):
     for date in np.unique(day):
      chosen=np.flatnonzero((rx==r)&(day==date))
      for row in permuted:row[chosen]=rng.permutation(y[chosen])
    def normalize(a):
     norm=np.sqrt((a*a).sum(1));return a/np.maximum(norm[:,None],1e-8),norm>1e-8
    def near(a,b):
     if a is None or b is None:assert a is None and b is None
     else:assert np.isclose(a,b,rtol=1e-9,atol=1e-10),(a,b)
    checked=0;null_checked=0;pair_checked=0
    for group in sizes:
     clean=features[('clean','raw',group)];unit,ref_valid=normalize(clean);selected=[]
     for r in np.unique(rx):
      for date in np.unique(day):
       for tx in range(6):
        choices=np.flatnonzero((rx==r)&(day==date)&(y==tx));selected.extend(sorted(choices,key=lambda i:str(ids[i]))[:2])
     selected=np.asarray(selected);domain={}
     # Independent:average clean unit vectors by balanced reference domain/class;
     # take query dot products,without a classifier decision or fitted head.
     for r in np.unique(rx):
      for date in np.unique(day):
       eligible=[];missing=0
       for rr in np.unique(rx):
        for dd in np.unique(day):
         if rr==r or dd==date:continue
         cell=selected[(rx[selected]==rr)&(day[selected]==dd)&ref_valid[selected]];bytx=[cell[y[cell]==tx] for tx in range(6)]
         if any(not len(a) for a in bytx):missing+=1;continue
         eligible.append(np.stack([unit[a].mean(0) for a in bytx]))
       domain[(r,date)]=(np.mean(eligible,axis=0) if eligible else np.zeros((6,unit.shape[1])),len(eligible),missing)
     for view in views:
      scores={};valid={};zero={};cells=np.zeros(n,int);missing=np.zeros(n,int);errors={}
      for name,a in [(name,features[(view,name,group)]) for name in readouts]+[('clean',clean)]:
       query,qvalid=normalize(a);values=np.zeros((n,6))
       for (r,date),(means,count,absent) in domain.items():
        mask=(rx==r)&(day==date);values[mask]=query[mask]@means.T;cells[mask]=count;missing[mask]=absent
       scores[name]=values;valid[name]=(cells>0)&qvalid;zero[name]=~qvalid
       difference=a-clean;errors[name]=(difference*difference).sum(1) if group=='combined' else (difference*difference).mean(1)
      def margins(name,labels=y):
       a=scores[name];p=a[np.arange(n),labels];return p,(a.sum(1)-p)/5
      for name in candidates:
       common=valid['raw']&valid[name]&valid['clean']
       rows=[r for r in s['records'] if r['candidate']==name and r['group']==group and r['view']==view];assert len(rows)==len(groups)
       for record in rows:
        mask=groups[(record['dimension'],record['value'])]
        for key,readout in [('baseline','raw'),('restored',name),('clean','clean')]:
         item=record[key];good=mask&valid[readout];count=int(good.sum());p,q=margins(readout)
         assert item['count']==int(mask.sum()) and item['valid_count']==count and item['zero_query_count']==int((mask&zero[readout]).sum())
         assert item['eligible_reference_cells_total']==int(cells[mask].sum()) and item['missing_reference_cells_total']==int(missing[mask].sum())
         for field,values in [('mean_same_TX_cosine',p),('mean_other_TX_cosine',q),('mean_margin',p-q)]:near(item[field],float(values[good].mean()) if count else None)
         near(item['positive_margin_percent'],100*float(((p-q)[good]>0).mean()) if count else None)
        good=mask&common;count=int(good.sum());p0,q0=margins('raw');p1,q1=margins(name);pc,qc=margins('clean')
        assert record['paired_delta']['count']==count
        near(record['paired_delta']['mean_margin_change'],float((p1-q1-p0+q0)[good].mean()) if count else None)
        near(record['paired_delta']['restored_minus_clean_margin'],float((p1-q1-pc+qc)[good].mean()) if count else None)
        assert record['paired_descriptor_error']['count']==int(mask.sum())
        near(record['paired_descriptor_error']['raw'],float(errors['raw'][mask].mean()));near(record['paired_descriptor_error']['restored'],float(errors[name][mask].mean()))
        checked+=1
       for label,readout in [('raw','raw'),('restored',name),('clean','clean')]:
        original=next(r for r in s['query_label_negative_controls'] if (r['candidate'],r['group'],r['view'],r['readout'])==(name,group,view,label))
        count=int(common.sum());assert original['count']==count and original['permutations']==199 and not original['is_p_value']
        if count:
         a=scores[readout][common];p=permuted[:,common];own=a[np.arange(count)[None,:],p];null=((6*own-a.sum(1))/5).mean(1)
         pos,neg=margins(readout);observed=float((pos-neg)[common].mean())
         for key,value in [('observed_mean',observed),('null_mean',float(null.mean())),('null_p05',float(np.quantile(null,.05))),('null_p95',float(np.quantile(null,.95)))]:near(original[key],value)
         assert original['above_null_p95']==bool(observed>float(np.quantile(null,.95))+1e-12)
        null_checked+=1
      common=valid['clean'].copy()
      for name in readouts:common&=valid[name]
      for a,b in d.COMPARISONS:
       record=next(r for r in s['comparisons'] if (r['candidate'],r['control'],r['group'],r['view'])==(a,b,group,view));count=int(common.sum())
       pa,na=margins(a);pb,nb=margins(b);assert record['common_count']==count
       near(record['margin_change'],float((pa-na-pb+nb)[common].mean()) if count else None)
       near(record['descriptor_error_change'],float((errors[a][common]-errors[b][common]).mean()) if count else None);pair_checked+=1
      # Yield between fixed view/group units so pending native source/target scorers can proceed.
      fcntl.flock(lock,fcntl.LOCK_UN)
      time.sleep(.1)
      fcntl.flock(lock,fcntl.LOCK_EX)
    assert checked==4536 and null_checked==756 and pair_checked==392
    reference=read(cached/'summary.json')
    original={(r['candidate'],r['group'],r['view'],r['dimension'],r['value']):r for r in reference['records']}
    cached_checked=0
    for row in s['records']:
     if row['candidate'] in ('r46_350521','r49_retain_control'):
      old=original[(d.CACHED_READOUTS[row['candidate']],row['group'],row['view'],row['dimension'],row['value'])]
      assert {k:v for k,v in row.items() if k!='candidate'}=={k:v for k,v in old.items() if k!='candidate'}
      cached_checked+=1
    assert cached_checked==1008
    logs={p.name:p.read_text(errors='replace') for p in (root/'logs').glob('*.log')};logs['controller']=(base/'logs'/root.name/'runner.log').read_text(errors='replace')
    assert all('Traceback (most recent call last)' not in value for value in logs.values())
    events=[json.loads(line) for line in logs['extract.log'].splitlines() if line.startswith('{')]
    assert [(r['view'],r['readout']) for r in events]==[(view,name) for view in views for name in readouts]
    timing=dict(extract_process_seconds=launches['score']['started']-launches['extract']['started'],
     score_process_seconds=(root/'complete.json').stat().st_mtime-launches['score']['started'],total_seconds=(root/'complete.json').stat().st_mtime-launch['started'])
    print(json.dumps(dict(status='VERIFIED',scope='4536formal_group_records,756conditionalnegativecontrols,392pairedchanges independently recomputed;not per-detail TX proof',
     summary=s,complete=done,feature_complete=meta,calibration_reference=cal,config=config,launch=launch,worker_launches=launches,processes=processes,
     logs=logs,timing=timing,independent_groups_checked=checked,independent_negative_controls_checked=null_checked,independent_comparisons_checked=pair_checked,
     cached_baseline_records_unchanged=cached_checked,goal_complete=False)))

if __name__=='__main__':main()
