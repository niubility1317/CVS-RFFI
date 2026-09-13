"""Observed activation states. Configuration is never reported as gradient evidence."""
from collections import Counter


class ActivationLedger:
    def __init__(self,row):
        self.row=row;self.steps=0;self.counts={};self.actions=Counter();self.selections=Counter()

    def observe(self,record):
        if record['step']!=self.steps:raise ValueError('activation ledger step gap')
        self.steps+=1;self.actions[record['action']]+=1
        dr=record.get('daot_rc4',{})
        for key in ('hard_count','partial_count','negative_count','satellite_samples'):
            self.selections[key]+=dr.get(key,0)
        weighted=dict(dr.get('weighted_components',{}))
        if self.row['x_enabled']:weighted['x']=record['terms'].get('x_cross_rx',0)*self.row['lambda_x']
        if self.row['u_enabled']:weighted['ux_normalized']=record['terms'].get('u_normalized',0)*self.row['lambda_u_interaction']
        for key,value in weighted.items():
            c=self.counts.setdefault(key,dict(observed_records=0,nonzero_loss_records=0,gradient_probes=0,nonzero_gradient_probes=0))
            c['observed_records']+=1;c['nonzero_loss_records']+=abs(value)>1e-12
            grad=dr.get(key+'_grad_norm')
            if key=='x':grad=record['fusion'].get('x_weighted_identity_grad_norm')
            if key=='ux_normalized':grad=record['fusion'].get('u_weighted_identity_grad_norm')
            if grad is not None:
                c['gradient_probes']+=1;c['nonzero_gradient_probes']+=grad>1e-12

    def report(self,epoch):
        parts={}
        for key,c in self.counts.items():
            start=61 if key=='daot_tangent' else (21 if key.startswith('daot_') or key in ['rc4_hard','rc4_partial_set','rc4_partial_conditional','rc4_negative','rc4_satellite','rc4_anchor'] else 1)
            state='NONZERO_GRADIENT_OBSERVED' if c['nonzero_gradient_probes'] else ('NONZERO_LOSS_GRADIENT_UNPROVEN' if c['nonzero_loss_records'] else ('WARMUP_NOT_DUE' if epoch<start else 'ENABLED_NO_NONZERO_EVIDENCE'))
            parts[key]=dict(configured=True,scheduled_start_epoch=start,status=state,**c)
        return dict(schema='xuc.observed_activation.v1',row=self.row['id'],steps=self.steps,epoch=epoch,
                    components=parts,selected_sample_occurrences=dict(self.selections),actions=dict(self.actions),
                    cstar_configured=self.row.get('cstar_action_enabled',False),
                    cstar_actual_control_actions=(self.actions['CATCHUP']+self.actions['CORRECT']) if self.row.get('cstar_action_enabled',False) else 0,
                    target_metrics_used=False)
