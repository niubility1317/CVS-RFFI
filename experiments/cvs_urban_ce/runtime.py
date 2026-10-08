"""Process-local training augmentation; validation and prediction stay unchanged."""
from contextlib import contextmanager
import csv
import json
from pathlib import Path
from experiments.cvs_phase1_stack.runtime import installed as base_installed
from experiments.cvs_urban_ce.augmentation import bounded_echo,satellite_keep

@contextmanager
def installed(c):
    with base_installed(c) as n:
        prepare=n._prepare_concat_sat_batch_for_training;detach=n._detach_log_mapping
        last={};epoch_totals={};current={}
        def flush():
            if not epoch_totals:return
            row=dict(epoch=current['epoch'],**epoch_totals)
            out=Path(current['output']);out.mkdir(parents=True,exist_ok=True)
            with (out/'channel_epoch.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            path=out/'channel_epoch.csv';fresh=not path.exists()
            with path.open('a',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row))
                if fresh:w.writeheader()
                w.writerow(row)
            print('[CHANNEL] '+json.dumps(row),flush=True)
            epoch_totals.clear()
        def augmented(aug,x,y,d,*,args,epoch,batch_idx):
            if current.get('epoch')!=epoch:flush()
            current.update(epoch=epoch,output=args.output_dir)
            last.clear()
            if c['receiver_dg']:
                x,stats=bounded_echo(x,seed=c['augmentation_seed'],epoch=epoch,step=batch_idx,
                    probability=c['dg_probability'],maximum=c['echo_l1_max'],ramp_epochs=c['dg_ramp_epochs'],delays=c['echo_delays'])
                last.update(stats)
            q=c['satellite_compute_probability']
            keep=satellite_keep(c['augmentation_seed'],epoch,batch_idx,q) if q<1 else True
            # Thin only the auxiliary satellite branch; every step keeps L CE.
            # Skip unused satellite computation before its E80 loss starts.
            omit=c['arm']=='urban_dg_thin' and (epoch<80 or not keep)
            args.concat_sat_ce_weight=(1./q) if not omit else 0.
            result=prepare(None if omit else aug,x,y,d,args=args,epoch=epoch,batch_idx=batch_idx)
            last.update(satellite_retained=int(result[3] is not None),satellite_importance=args.concat_sat_ce_weight,
                        model_input_samples=len(x)*(2 if result[3] is not None else 1))
            for key,value in dict(steps=1,dg_selected=last.get('dg_selected',0),source_samples=len(x),
                                  satellite_branches=last['satellite_retained'],model_input_samples=last['model_input_samples']).items():
                epoch_totals[key]=epoch_totals.get(key,0)+value
            return result
        def log(values):
            if 'train/loss' in values:
                values=dict(values,**{'train/channel_'+k:v for k,v in last.items()})
            return detach(values)
        n._prepare_concat_sat_batch_for_training=augmented;n._detach_log_mapping=log
        try:yield n
        finally:
            flush()
            n._prepare_concat_sat_batch_for_training=prepare;n._detach_log_mapping=detach
