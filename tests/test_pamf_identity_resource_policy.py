from experiments.pamf_identity_flow.resource_policy import is_restoration_gpu_worker


def test_other_restoration_gpu_entries_and_cpu_scorers():
    cases={
        '100 python -m experiments.pamf_phase3_restoration.train --root /x':True,
        '101 python -m experiments.pamf_phase3_restoration.historical_eval predict --root /x':True,
        '102 python -m experiments.pamf_phase3_restoration.phase3_eval predict --config /x':True,
        '103 python -m experiments.pamf_phase3_restoration.cache worker --root /x':True,
        '104 python -m experiments.pamf_phase3_restoration.cost_probe worker --root /x':True,
        '105 python /release/experiments/pamf_phase3_restoration/train.py --root /x':True,
        '106 python -m experiments.pamf_phase3_restoration.historical_eval score --root /x':False,
        '107 python -m experiments.pamf_phase3_restoration.cache run --root /x':False,
        '108 python -m unrelated.train --root /x':False,
    }
    for command,expected in cases.items():assert is_restoration_gpu_worker(command)==expected
