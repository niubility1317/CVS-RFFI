"""Six authorized ratio rows; all scientific options inherited except concat scenes."""
from pathlib import Path
from copy import deepcopy
import json

ROOT=Path(__file__).resolve().parents[1]
RUN='20260920-phase1-daot-rc4-residual-ratios-manysig-s392005-r01'
SCENES=['clean','practical_high','practical_mid','practical_low_urban','practical_low_suburban','practical_mid_urban','practical_high_urban']
ACCEL={'--practical_execution_fast':'true','--practical_buffer_input':'true',
    '--practical_buffer_output':'true','--practical_channel_workers':'2',
    '--source_validation_reuse':'true','--incremental_epoch_telemetry':'true',
    '--practical_eval_prefetch':'true','--practical_eval_cpu_pipeline':'true',
    '--a1_eval_identity_only':'true','--a1_gradient_snapshot':'true',
    '--practical_eval_cache_dir':f'/home/szu2070436088/2510044040/CV-SincNet/runs/{RUN}/evaluation_view_cache'}

def main():
    base=json.loads((ROOT/'configs/rc4_practical_residual_noeq_20260918.json').read_text(encoding='utf-8'))
    rows=[]
    for easy in ('high','mid'):
        for e,h in ((3,1),(2,2),(1,3)):
            name=f'RESIDUAL_{easy.upper()}_LOWURBAN_{e*25}_{h*25}_s392005'
            doc=deepcopy(base);doc['id']=name
            opts=doc['options'];tokens=f'practical_{easy}*{e},practical_low_urban*{h}'
            opts.update(ACCEL)
            opts.update({'--sat_train_scenario':f'practical_{easy}',
                '--sat_train_scenarios':f'practical_{easy},practical_low_urban',
                '--sat_view_schedule':';'.join(f'{start}@{prob}:{tokens}' for start,prob in ((1,'0.30'),(41,'0.60'),(91,'0.80'))),
                '--a1_periodic_target_scenarios':','.join(SCENES)})
            filename=f'rc4_residual_ratio_{easy}_{e*25}_{h*25}_20260920.json'
            (ROOT/'configs'/filename).write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
            rows.append(dict(id=name,config=filename,easy=f'practical_{easy}',hard='practical_low_urban',
                easy_probability=e/4,hard_probability=h/4,ratio_scope='applied labeled concat satellite views only'))
    matrix=dict(run_id=RUN,rows=rows,scenes=SCENES,seed=392005,epochs=200,
        base_recipe='rc4_practical_residual_noeq_20260918.json',
        daot_internal_views='unchanged: student high, teacher hard mid and low_urban',
        initialization='scratch; old checkpoints evaluation only')
    (ROOT/'configs/practical_ratio_matrix_20260920.json').write_text(json.dumps(matrix,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
