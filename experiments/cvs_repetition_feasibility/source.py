"""Full original L6300 observation feasibility; no network or checkpoint used."""
import json,os,sys,time
from pathlib import Path
import torch
from .objective import repetition_residuals
from .metrics import summarize,qualification
from .relations import measure

RUN='20261014-phase1-visible-period-sourceL-manysig-m1-r01'
RELEASE='cvs_visible_period_sourceL_20261014_r01'
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'

def run(request):
    expected=PROJECT+'/runs/'+RUN+'/L6300/source'
    if request['run_id']!=RUN or request['output_root']!=expected or request['source_role']!='L_s':
        raise ValueError('Only fixed source-L6300 diagnostic and owned output root')
    out=Path(expected)
    if out.exists():raise FileExistsError('Preserve existing source diagnostic; no relaunch')
    torch.set_num_threads(2);started=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    runtime=dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,device='cpu',GPU_allocations=0,
                 analysis_code_commit=request['analysis_code_commit'],run_id=RUN)
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    print('RUNTIME '+json.dumps(runtime),flush=True)
    # This stage consumes no checkpoint/network: no-query fixture immediately
    # continues into legal source observations, with no authorization token.
    smoke=repetition_residuals(torch.zeros(2,2,256))
    if smoke['valid'].any() or smoke['target'].shape!=(2,24) or any(p.any() for p in smoke['predictions'].values()):
        raise ValueError('No-data observation primitive smoke failed')
    from experiments.cvs_pa_recomposition import design as d
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    c=d.config(next(r for r in d.rows() if r['row_id']=='ZERO-s2026092701'))
    predictions={k:[] for k in ('copy_cycle1','visible_mean','known_reference')}
    targets=[];valid=[];labels=[];receivers=[];days=[];physical_ids=[]
    with installed(c) as native:
        context=native._build_ssdg_wisig_data(configure(d.make_args(c,'cpu')),torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context)!=d.read(d.SOURCE)['role_ids']:
            raise ValueError('Original exact source-only data roles required')
        dataset=context['train_loader'].dataset
        if len(dataset)!=6300 or getattr(dataset,'hide',True):raise ValueError('Original visible source L6300 only')
        with torch.no_grad():
            for begin in range(0,6300,128):
                samples=[dataset[i] for i in range(begin,min(begin+128,6300))]
                x=torch.stack([s[0] for s in samples])
                result=repetition_residuals(x)
                if result['summary']['calibration_uses_holdout'] or result['masked_input'][:,:,120:140].any():
                    raise ValueError('Hidden observation entered calibration or masked input')
                valid.append(result['valid']);targets.append(result['target'])
                for arm,p in result['predictions'].items():predictions[arm].append(p)
                # Source labels are first used after every prediction is fixed.
                for _,y,_,meta in samples:
                    labels.append(int(y));receivers.append(int(meta['rx_i']));days.append(int(meta['day_i']))
                    physical_ids.append([int(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')])
                if begin%1024==0:print(f'SOURCE_L_SEEN {min(begin+len(samples),6300)}/6300',flush=True)
    target=torch.cat(targets);valid=torch.cat(valid);predictions={k:torch.cat(v) for k,v in predictions.items()}
    labels=torch.tensor(labels);receivers=torch.tensor(receivers);days=torch.tensor(days)
    metrics=summarize(valid,target,predictions,labels,receivers,days)
    opaque_ids=[json.dumps(v,separators=(',',':')) for v in physical_ids]
    geometry=measure(predictions['visible_mean'],labels.tolist(),receivers.tolist(),days.tolist(),opaque_ids,valid,role='L_s')
    summary=dict(status='SOURCE_OBSERVATION_FEASIBILITY_COMPLETE',run_id=RUN,analysis_code_commit=request['analysis_code_commit'],
                 source_role='L_s',source_count=6300,source_IQ_packet_reads=6300,network_IQ_forwards=0,
                 noquery_fixture_packets=2,checkpoint_sources=[],model_or_decoder_fitting=False,
                 head_fit_updates=0,optimizer_updates=0,EMA_updates=0,V_gradient_updates=0,U_IQ_reads=0,
                 U_truth_read=False,target_access=False,metrics=metrics['records'],metric_normalization=metrics['normalization'],
                 relation_summary=geometry['summary'],qualification=qualification(metrics,geometry['summary']),
                 primitive=dict(holdout=[120,140],visible=[[80,100],[100,120],[140,160]],reference_delay_grid=320,
                                relative_visible_gain_threshold=1e-12,masked_holdout_fill='zero'),
                 runtime=runtime,wall_seconds=time.perf_counter()-started,
                 limitation='Seen-source received/equalized reference residual24; not deployed joint160, identifiable TX hardware, raw-noise independence, trained decoder benefit, hiddenU labels or DG/Phase2 performance.')
    torch.save(dict(target=target,predictions=predictions,valid=valid,labels=labels,receivers=receivers,days=days,
                    physical_ids=physical_ids,packet_target_energy=metrics['packet_target_energy'],packet_MSE=metrics['packet_MSE'],
                    relations=geometry['tensors']),out/'observation_census.pt')
    d.write(out/'summary.json',summary)
    print('SOURCE_OBSERVATION_FEASIBILITY_COMPLETE '+json.dumps(dict(qualification=summary['qualification'],overall=summary['metrics']['overall'],geometry=summary['relation_summary']['overall'])),flush=True)
    return summary
