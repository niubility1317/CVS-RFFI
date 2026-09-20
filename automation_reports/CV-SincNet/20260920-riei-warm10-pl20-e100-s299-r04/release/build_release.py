from pathlib import Path
import json
P=Path(__file__).resolve().parent;OLD=P.parent/'20260920-riei-original-concat-pl-s299-e200-r03'
for n in ['train.py','dispatch.py','remote_action.py','deliver.py','get_dataset_lab_unlab.py','Extracter_and_Classifier.py',
          'complexcnn.py','optimized_training.py','riei_model.py','riei_architecture.py','riei_losses.py']:
    (P/n).write_bytes((OLD/n).read_bytes())
(P/'satellite_channel.py').write_bytes((P.parent/'satellite_channel.py').read_bytes())
p=P/'train.py';s=p.read_text(encoding='utf-8')
s=s.replace('from get_dataset_lab_unlab import TrainDataset, TestDataset','from get_dataset_lab_unlab import TestDataset\nfrom source_data import load_source,warm_view')
s=s.replace('pseudo_start=100):','pseudo_start=20):').replace('if epoch>=pseudo_start:', 'if xu is not None and epoch>=pseudo_start:')
s=s.replace('pseudo_active=epoch>=pseudo_start','pseudo_active=xu is not None and epoch>=pseudo_start')
s=s.replace('unlabeled_views=2*len(xu)','unlabeled_views=0 if xu is None else 2*len(xu)')
s=s.replace("    parser.add_argument('--output',required=True)","    parser.add_argument('--output',required=True)\n    parser.add_argument('--use-pl',type=int,choices=[0,1],required=True)")
s=s.replace('method=args.method,pid=os.getpid()', 'method=args.method,use_pl=bool(args.use_pl),pid=os.getpid()')
a=s.index('    data=TrainDataset(');b=s.index("    arrays={'source_clean'",a)
s=s[:a]+"""    xl,yl,xv,yv,xu,yu=load_source(args.data_root,out/'splits',bool(args.use_pl))
    assert set(yl[:,2])==set([0,1,2,3,5])
    if xu is not None:assert 4 not in set(yu[:,2]) and (yu[:,0]==-1).all()
    warm=xl.copy();warm[:,1]=warm_view(xl[:,0])
"""+s[b:]
a=s.index('    label_dataset=TensorDataset(');b=s.index('    model=create_model(',a)
s=s[:a]+"""    label_dataset=TensorDataset(tensor(xl),tensor(yl))
    warm_dataset=TensorDataset(tensor(warm),tensor(yl))
    label_generator=torch.Generator()
    ld=DataLoader(label_dataset,batch_size=64,shuffle=True,generator=label_generator)
    wd=DataLoader(warm_dataset,batch_size=64,shuffle=True,generator=label_generator)
    ud=None
    if args.use_pl:
        ud=DataLoader(TensorDataset(tensor(xu),tensor(yu)),batch_size=256,shuffle=True,generator=torch.Generator())
    write_json(out/'data_summary.json',dict(labeled=len(xl),unlabeled_loaded=0 if xu is None else len(xu),
        validation=len(xv),source_rxs=[0,1,2,3,5],target_rx=4,use_pl=bool(args.use_pl),steps_per_epoch=len(ld),
        no_pl_does_not_index_unlabeled_iq=not bool(args.use_pl)))
"""+s[b:]
s=s.replace('for epoch in range(1,201):','for epoch in range(1,101):')
a=s.index('        for step,((bl,by),(bu,bd))');b=s.index('            for key,value in metrics.items()',a)
s=s[:a]+"""        label_generator.manual_seed(299+epoch)
        active_loader=wd if epoch<=10 else ld
        ui=None
        if ud is not None and epoch>=20:
            ud.generator.manual_seed(1299+epoch);ui=iter(ud)
        for step,(bl,by) in enumerate(active_loader):
            bl=bl.to(device);bu=bd=None
            if ui is not None:
                try:bu,bd=next(ui)
                except StopIteration:ui=iter(ud);bu,bd=next(ui)
                bu=bu.to(device);bd=bd[:,1].to(device)
            metrics=train_step(model,args.method,optimizers,bl,by.to(device),bu,bd,epoch,
                               feature_norm=.0001,pseudo_start=20)
            append(out/'batches.jsonl',dict(epoch=epoch,batch=step,**metrics,
                augmentation='moderate_Loo_sigma1' if epoch<=10 else 'original_fixed_Loo_sigma3',
                loo_sigma=1. if epoch<=10 else 3.))
"""+s[b:]
s=s.replace("epoch=200,method=args.method", "epoch=100,method=args.method,use_pl=bool(args.use_pl)")
p.write_text(s,encoding='utf-8',newline='\n')
m=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'));remote=m['remote_root'].replace(OLD.name,P.name)
m.update(group_id=P.name,remote_root=remote,epochs=100,pseudo_start_epoch=20,rows=[],replaces_run_id=None,
    comparison_scope='Simple RIEI: moderate Loo E1:10, original fixed Loo E11:100; PL20 vs no U',
    augmentation=dict(warm_epochs=[1,10],warm_loo_delta2=1.,warm_seed=340299,scatter_variance=.0049,
                      full_epochs=[11,100],full_loo_delta2=9.,test='unchanged original fixed Loo',extra_cfo=False,extra_iq=False),
    inference=dict(selection='E100 student only',ema=False),initialization='scratch both rows; no inherited weights')
m['pseudo'].update(start_epoch=20,weight=.65)
for use in [0,1]:
    run=f'riei-warm10-{"pl20" if use else "no-unlabeled"}-s299-e100-r04'
    m['rows'].append(dict(method='riei',use_pl=use,run_id=run,output_root=remote+'/runs/'+run))
(P/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
p=P/'dispatch.py';s=p.read_text(encoding='utf-8').replace("get('epoch')==200","get('epoch')==100")
s=s.replace("'--data-root',m['data_root'],'--output',r['output_root']]", "'--data-root',m['data_root'],'--output',r['output_root'],'--use-pl',str(r['use_pl'])]")
p.write_text(s,encoding='utf-8',newline='\n')
spec=json.loads((OLD/'registry_spec.json').read_text(encoding='utf-8'))
spec.update(run_id=P.name,group_id='riei-simple-warm10-pl-vs-no-u',display_name='RIEI前10轮中等Loo增强：PL20与完全无U两组100轮',
    description=m['comparison_scope'],replaces_run_id=None,status='PREPARED')
spec['code'].update(commit=None,cwd=remote)
spec['checkpoint'].update(contract_check_ref=str(P/'report.md'),selection_rule='fixed E100 student; no EMA')
spec['data']['contract_ref']=str(P/'manifest.json');spec['data']['leo_config_ref']=str(P/'manifest.json')
for k,v in spec['execution'].items():
    if isinstance(v,str):spec['execution'][k]=v.replace(OLD.name,P.name)
prototype=spec['rows'][0];spec['rows']=[]
for row in m['rows']:
    r=dict(prototype);r.update(row_id=row['run_id'],method='RIEI+PL20' if row['use_pl'] else 'RIEI no unlabeled samples',
        epochs=100,config_ref=str(P/'manifest.json'),resolved_config_ref=row['output_root']+'/resolved_config.json',
        output_root=row['output_root'],log_path=remote+'/logs/'+row['run_id']+'.log',
        budget_ref='E100 labeled64 both groups; U256 only PL group E20+',
        data_overrides=dict(use_unlabeled=bool(row['use_pl'])),
        command=['python','train.py','--method','riei','--use-pl',str(row['use_pl']),'--data-root',m['data_root'],'--output',row['output_root']])
    spec['rows'].append(r)
spec['notes']=['Old200 logs and frozen predictions diagnosed, no evidence of numerical failure.',
    'No-U loader reads labels for identical split but never indexes unlabeled IQ; no U forward/loss/BN update.',
    'Original RIEI optimizer/losses unchanged; warm10 is sole shared training change. Test fixed full-strength remains unchanged.']
(P/'registry_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
print('BUILT_TWO_ROWS')
