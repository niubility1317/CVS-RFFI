"""Source-only reconstruction-tail loss prototype; no real run is launched."""
from experiments.pamf_r46_weight_mean import design as base

RUN='20261021-phase1-restoration-R46-reconstruction-tail-s2026092701-m2-r68'
ARMS=('tail_half','tail_quarter')
FRACTION=dict(tail_half=.5,tail_quarter=.25)
MIX=.5
PARAMETERS=273765
CONTEXT_PARAMETERS=185615
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT,base.COMPACT_PARENT_ROW
CONTROL=dict(run=base.RUN,commit='21141083f91ca7703b9242e5d90873666a8414b6',row='final',parameters=PARAMETERS,
    role='Read-only future completed mean-objective control;never initialization or teacher')
teacher_weight=base.teacher_weight


def configuration():
    return dict(schema='R46_source_reconstruction_tail_prototype_v1',run=RUN,stage='SYNTHETIC_PROTOTYPE_ONLY',
        arms=list(ARMS),fraction=FRACTION,mix=MIX,parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        architecture='Existing273765staticGN4U-Net/3tiedpriors/6CG graph;no new restorer family',
        initialization='Only direct original humanM35,original staticGN4 weight-only maps',teacher='Original frozen humanM35 only',
        loss='0.5mean(reconstruction)+0.5mean(topceil(q*B)reconstruction)+lambda(epoch)mean(original3stageIQteacher)',
        reconstruction='pairedclean IQNMSE+.05multiscale relativepower spectral error',
        selection='Detached current source reconstruction error only;stable descending sort;no TX/RX/F/V/target or persistent hardness state',
        all_examples='Every L/U example retains weight>=.5;reconstruction weights sum to current batch size;teacher term unchanged and all examples participate',
        comparison='q=.5 versusq=.25;original mean control R66final is fixed before any R66 recognition accuracy access',
        control=CONTROL,epochs=20,updates=4940,batch=256,microbatch=256,source_examples=1260000,L_exposures=126000,U_exposures=1134000,
        new_trainable_parameters=0,new_calibration_IQ=0,extra_training_G_forwards=0,extra_inference_operations=0,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,
        actual_source_epochs=0,actual_optimizer_updates=0,real_IQ_reads=0,GPU_launches=0,
        predecessor='All same-card R66 readout workers must complete and actually exit before any future GPU use',
        future_test='Both fixed versions auto source1800x7/profile/formal168000x7;all seven predictions before independent truthlast;no source gate',
        known_related_work='Existing pamf_hardness_refine uses F-semantic loss and different parents;only the generic tail-aggregation idea is related,no weights/F/semantic loss inherited',
        hypothesis='Worst-source-reconstruction focus may change restoration stability;may chase noise or reduce other views;no recognition/identity/latency gain presumed',
        goal_complete=False)
