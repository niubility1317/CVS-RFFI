import json
from pathlib import Path

here = Path(__file__).parent
probe = json.loads((here / 'phase1_source_expert_router_probe.json').read_text(encoding='utf-8'))
replay = json.loads((here / 'phase1_source_expert_router_replay.json').read_text(encoding='utf-8'))
assert probe['source_ids'] == probe['train_ids'] + probe['validation_ids'] + probe['heldout_ids'] == 4096
assert probe['source_L_s_ids'] + probe['source_U_s_ids'] == 4096
assert probe['training_source_valid_views'] == 7194
assert probe['training_repairable_views'] == 120
assert probe['experts'] == ['raw', 'dura', 'boundary', 'feature']
assert probe['U_s_labels_read'] == probe['V_reads'] == probe['test_IQ_reads'] == 0
assert probe['expert_decisions_before_source_labels']
assert replay['status'] == 'VERIFIED_SOURCE_ROUTER_REPLAY' and replay['models_refit'] == 0
assert replay['U_s_labels_read'] == replay['V_reads'] == replay['test_IQ_reads'] == 0
assert [r['view'] for r in replay['results']] == ['clean', '01', '20']
for original, result in zip(probe['heldout'], replay['results']):
    assert original == result['router']
    assert original['valid_count'] == original['L_s_count'] + original['U_confident_count']
    assert sum(original['expert_choices'].values()) == original['valid_count']
    assert original['router'] == original['baseline'] + original['repairs'] - original['damages']
    oracle = result['oracle']
    assert oracle['baseline'] == original['baseline']
    assert oracle['four_expert_oracle'] == oracle['baseline'] + oracle['available_rescues']
    assert oracle['four_expert_oracle'] == oracle['three_expert_oracle'] + oracle['raw_unique']
print('VERIFIED: source splits, provenance flags, replay, paired counts, and restricted oracle identities')
