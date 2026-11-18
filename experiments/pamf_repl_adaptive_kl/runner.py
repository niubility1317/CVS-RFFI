"""Run clean-only fixed-anchor comparisons with adaptive U sample weights."""
from experiments.pamf_repl_view_dose.runner import main
from . import plan
from .objective import installed


if __name__ == '__main__':
    main(plan, 'experiments.pamf_repl_adaptive_kl.runner', objective=installed,
         max_total=2, reserve_mib=12288)
