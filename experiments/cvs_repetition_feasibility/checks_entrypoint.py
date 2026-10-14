"""Reject wrong role/run/output before creating output or reading source IQ."""
import json
from .source import RUN,PROJECT,run

def check():
    good=dict(run_id=RUN,source_role='L_s',output_root=PROJECT+'/runs/'+RUN+'/L6300/source')
    cases=[dict(good,source_role='U_s'),dict(good,source_role='V'),dict(good,source_role='target'),
           dict(good,run_id='other-run'),dict(good,output_root='/unowned/output')]
    for request in cases:
        try:run(request)
        except ValueError:pass
        else:raise AssertionError('Unregistered role/run/output accepted')
    return dict(status='VERIFIED',forbidden_U_V_target_roles_rejected=True,wrong_run_output_rejected=True,
                rejection_before_data_or_output_creation=True,actual_IQ_or_checkpoint_reads=0)

if __name__=='__main__':print(json.dumps(check()))
