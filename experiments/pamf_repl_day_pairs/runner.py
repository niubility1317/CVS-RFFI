"""Run the matched RX and source-date contrast comparisons."""
from experiments.pamf_repl_view_dose.runner import main
from . import plan
from .objective import installed


if __name__ == '__main__':
    main(plan, 'experiments.pamf_repl_day_pairs.runner', objective=installed,
         max_total=2, reserve_mib=12288)
