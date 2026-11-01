"""Build fixed low-SNR network replay without clean target or simulator SNR."""
from pathlib import Path

p = Path(__file__).parent
source = (p / 'verify_phase1_source_cycle_residual_net.py').read_text(encoding='utf8')
changes = [
    ('phase1_source_cycle_residual_net.json', 'phase1_source_cycle_low_snr_net.json'),
    ("result['selected']['kind'] == 'cycle_residual_net'", "result['selected']['kind'] == 'cycle_low_snr_net'"),
    ('phase1_source_cycle_residual_net.py', 'phase1_source_cycle_low_snr_net.py'),
    ("marker = \"train_input=torch.cat([data[v]['input'] for v in views])\"", "old_snr = \"snrs.append(float(meta['signal_to_added_noise_db']))\"\nassert inner.count(old_snr) == 1\ninner = inner.replace(old_snr, 'snrs.append(0.)')\nmarker = 'teacher.eval();teacher.requires_grad_(False);train_snr='"),
    ('paired_clean_targets_used_for_inference=False,predictions_before_source_tx_labels=True', 'paired_clean_targets_used_for_inference=False,simulator_snr_used_for_inference=False,predictions_before_source_tx_labels=True'),
    ("assert not replay['paired_clean_targets_used_for_inference'] and replay['test_IQ_reads'] == 0", "assert not replay['paired_clean_targets_used_for_inference'] and not replay['simulator_snr_used_for_inference'] and replay['test_IQ_reads'] == 0"),
    ('phase1_source_cycle_residual_net_replay.json', 'phase1_source_cycle_low_snr_net_replay.json'),
    ('VERIFIED fixed cycle network without clean inference inputs', 'VERIFIED fixed low-SNR network without clean or simulator-SNR inference inputs'),
]
for old, new in changes:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
dest = p / 'verify_phase1_source_cycle_low_snr_net.py'
assert not dest.exists()
compile(source, str(dest), 'exec')
dest.write_text(source, encoding='utf8', newline='\n')
print('CREATED fixed low-SNR network source replay', dest)
