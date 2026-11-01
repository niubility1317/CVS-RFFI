import json
from pathlib import Path

p=Path(__file__).parent
probe=json.loads((p/'phase1_source_noise_guard_router_probe.json').read_text(encoding='utf8'))
replay=json.loads((p/'phase1_source_noise_guard_router_replay.json').read_text(encoding='utf8'))
assert probe['status']=='SOURCE_ONLY_NOISE_GUARDED_ROUTER_PROBE'
assert probe['source_ids']==probe['train_ids']+probe['validation_ids']+probe['heldout_ids']==4096
assert probe['source_L_s_ids']+probe['source_U_s_ids']==4096 and probe['excluded_previous_source_ids']==8704
assert probe['U_s_labels_read']==probe['V_reads']==probe['test_IQ_reads']==0 and probe['expert_decisions_before_source_labels']
mild=('01','01_plus_15dB','20','20_plus_15dB')
eligible=[v for v in probe['validation'] if v['clean_net']>=-8.2 and min(v['per_view'][k] for k in mild)>=-4]
selected=probe['selected']
assert selected['kind']=='source_router' and selected['validation_net']==max([0]+[v['strong_net'] for v in eligible])
assert any(v['epoch']==selected['epoch'] and v['threshold']==selected['threshold'] and v['energy_limit']==selected['energy_limit'] and v['strong_net']==selected['validation_net'] for v in eligible)
assert replay['status']=='VERIFIED_SOURCE_ROUTER_REPLAY' and replay['models_refit']==0
assert replay['U_s_labels_read']==replay['V_reads']==replay['test_IQ_reads']==0
assert [x['view'] for x in replay['results']]==['clean','01','01_plus_15dB','01_plus_5dB','20','20_plus_15dB','20_plus_5dB']
for original,result in zip(probe['heldout'],replay['results']):
    assert original==result['router']
    assert original['valid_count']==original['L_s_count']+original['U_confident_count']
    assert sum(original['expert_choices'].values())==original['valid_count']
    assert original['router']==original['baseline']+original['repairs']-original['damages']
    o=result['oracle']
    assert o['baseline']==original['baseline'] and o['four_expert_oracle']==o['baseline']+o['available_rescues']
    assert o['four_expert_oracle']==o['three_expert_oracle']+o['raw_unique']
print('VERIFIED source splits, seven views, provenance flags, paired counts and restricted oracle identities')
