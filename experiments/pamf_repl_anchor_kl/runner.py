"""Run anchor KL with GPU 7 excluded and a four-worker resource ceiling."""
from experiments.pamf_repl_view_dose.runner import main
from . import plan
from .objective import installed


if __name__ == '__main__':
    main(plan, 'experiments.pamf_repl_anchor_kl.runner', objective=installed,
         max_total=4, reserve_mib=6000)
