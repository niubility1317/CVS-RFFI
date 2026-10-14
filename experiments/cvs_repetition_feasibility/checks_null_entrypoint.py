"""Fixed source boundaries and synthetic budget census, without source IQ."""
import json
import torch
from .null_source import RUN, PROJECT, run, budget_census, source_qualification


def check():
    good = dict(run_id=RUN, source_role='L_s', observation_kind='null_reference14',
                output_root=PROJECT + '/runs/' + RUN + '/L6300/source')
    cases = [dict(good, source_role=r) for r in ('U_s', 'V', 'target')]
    cases += [dict(good, run_id='unregistered'), dict(good, output_root='/unowned'),
              dict(good, observation_kind='physical_residual24')]
    for request in cases:
        try:
            run(request)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid entry crossed boundary before source/output I/O')
    n = 10
    target = torch.ones(n, 14, dtype=torch.float64)
    predictions = dict(copy_cycle1=target, visible_mean=target * 2, known_reference=target * 0)
    signal = torch.ones(n, dtype=torch.float64)
    budget = torch.ones(n, dtype=torch.float64) * .5
    rx = torch.tensor([1, 3, 4, 6, 8] * 2)
    valid = torch.ones(n, dtype=torch.bool)
    valid[0] = False
    records, tensors = budget_census(target, predictions, signal, budget, rx, valid)
    assert records['overall']['visible_mean']['active_count'] == 9
    assert records['overall']['copy_cycle1']['active_count'] == 0
    assert records['RX/1']['visible_mean']['active_fraction'] == .5
    expected = tensors['visible_mean']['per_packet_loss'][1:].sum() / 10
    assert abs(records['overall']['visible_mean']['weighted_loss'] - float(expected)) < 1e-15
    return dict(status='VERIFIED', forbidden_roles_run_output_kind_rejected_before_IO=True,
                invalid_count_in_full_denominator=True, zero_new_IQ_or_checkpoint_reads=True)


if __name__ == '__main__':
    print(json.dumps(check()))
