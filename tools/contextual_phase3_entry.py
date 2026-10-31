"""Load a Git-pinned Phase3 evaluator alongside its frozen historical release."""
from pathlib import Path
import runpy

import experiments.pamf_phase3_open_world as phase3

# The historical package is complete; the new release contains only this
# evaluator. Extend its search path without copying or mutating the old release.
overlay = Path(__file__).resolve().parents[1]/'experiments/pamf_phase3_open_world'
phase3.__path__.insert(0, str(overlay))
runpy.run_module('experiments.pamf_phase3_open_world.contextual_restore_eval', run_name='__main__')
