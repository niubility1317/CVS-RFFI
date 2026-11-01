"""Verify fresh-source exclusions, fixed assets and inference-SNR isolation."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

p = Path(__file__).parent
spec = json.loads((p / 'phase1_source_gated_low_snr_spec.json').read_text(encoding='utf8'))
original = json.loads((p / 'phase1_source_gated_low_snr_probe.json').read_text(encoding='utf8'))
isolated = json.loads((p / 'phase1_source_gated_low_snr_isolation.json').read_text(encoding='utf8'))
for name, key in ((spec['script'], 'script_sha256'), (spec['model_source'], 'model_source_sha256')):
    assert sha256((p / name).read_bytes()).hexdigest() == spec[key]
assert spec['status'] == 'SOURCE_DIAGNOSTIC_FIXED_BEFORE_EXECUTION'
assert (original['source_ids'], original['validation_ids'], original['heldout_ids']) == (1622, 1117, 505)
assert original['predictions_before_source_tx_labels'] and not original['clean_target_used_for_inference'] and not original['simulated_snr_used_for_inference']
assert original['U_s_labels_read'] == original['V_reads'] == original['test_IQ_reads'] == 0
def without_reference_confusion(x):
    y = deepcopy(x)
    for group in (y['validation'], y['heldout'], *y['validation_groups'].values()):
        for row in group.values():
            del row['blind_snr_confusion']
    return y
assert without_reference_confusion(original) == without_reference_confusion(isolated)
for group in (original['validation'], original['heldout'], *original['validation_groups'].values()):
    for row in group.values():
        c = row['blind_snr_confusion']
        assert c['true_positive'] + c['false_negative'] == c['true_low']
        assert c['true_positive'] + c['false_positive'] + c['false_negative'] + c['true_negative'] == row['count']
assert original['heldout']['clean']['gate_enabled'] == 0
result = dict(status='VERIFIED_FRESH_SOURCE_GATED_LOW_SNR_ISOLATION', source_ids=original['source_ids'], validation_ids=original['validation_ids'], heldout_ids=original['heldout_ids'], script_sha256=spec['script_sha256'], model_sha256=spec['model_source_sha256'], all_nonreference_results_identical=True, simulator_snr_replaced_by_zero_in_replay=True, paired_clean_target_removed_from_inference=True, test_IQ_reads=0)
dest = p / 'phase1_source_gated_low_snr_verification.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED fresh-source fixed assets and simulator-SNR isolation', dest)
