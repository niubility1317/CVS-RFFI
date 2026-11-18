"""Run the fixed-view auxiliary dose matrix."""
from experiments.pamf_repl_view_dose.runner import main
from . import plan


if __name__ == '__main__':
    main(plan, 'experiments.pamf_repl_aux_view_dose.runner')
