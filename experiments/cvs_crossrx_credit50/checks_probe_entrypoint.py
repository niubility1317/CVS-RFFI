"""Source probe boundary rejects before any filesystem/checkpoint/data I/O."""
import json
from .source_probe import RUN,PROJECT,run

def check():
    good=dict(run_id=RUN,source_role='L_s',model_seed=2026092701,
              source_release=PROJECT+'/releases/cvs_feature_sequence50_20261013_r01',
              output_root=PROJECT+'/runs/'+RUN+'/SIG_TIME-s2026092701/source_diagnostic')
    for request in [dict(good,source_role=r) for r in ('U_s','V','target')]+[
        dict(good,run_id='other'),dict(good,output_root='/unowned'),dict(good,model_seed=2026092702),
        dict(good,source_release='/other-release')]:
        try:run(request)
        except ValueError:pass
        else:raise AssertionError('Invalid source scope crossed input/output boundary')
    return dict(status='VERIFIED',role_run_output_seed_rejected_before_IO=True,
                actual_data_or_checkpoint_reads=0)

if __name__=='__main__':print(json.dumps(check()))
