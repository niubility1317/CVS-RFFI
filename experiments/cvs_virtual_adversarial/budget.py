from pathlib import Path

def expected_counts(c):
 epochs=c['epochs'];updates=epochs*c['steps_per_epoch']
 if not 1<=epochs<=50 or c['updates']!=updates:raise ValueError('Invalid chosen epoch/update budget')
 active=max(0,updates-c['semi']['warmup_updates']) if c['source_U'] else 0
 return dict(identity_updates=updates,L_packets=updates*c['batch_size'],U_packets=active*c['batch_size'],accepted_U=active*c['batch_size'],teacher_forwards=active,search_forwards=active,search_grad_calls=active,consistency_forwards=active)

def final_artifacts(c,model,counts,distinct_U,elapsed_seconds,parameters):
 expected=expected_counts(c)
 if any(counts.get(k)!=v for k,v in expected.items()) or distinct_U!=(56700 if c['source_U'] else 0):raise ValueError('Fixed final source budget differs')
 ck=dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,epoch=c['epochs'],updates=c['updates'],ancestors=[],initialization='scratch',target_access=False,checkpoint_selection=c['checkpoint_selection'])
 done=dict(status='SOURCE_TRAINED',config=c,epoch=c['epochs'],updates=c['updates'],logged_steps=c['updates'],counts=counts,distinct_U=distinct_U,target_access=False,U_truth_read=False,ancestors=[],elapsed_seconds=elapsed_seconds,parameters=parameters)
 frozen=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(Path(c['output'])/'identity.pth'),epoch=c['epochs'],updates=c['updates'],selection=c['checkpoint_selection'],target_access=False)
 return ck,done,frozen
