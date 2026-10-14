"""One immutable fixed null-budget source-only release."""
from .publish import main

if __name__ == '__main__':
    main(run_id='20261014-phase1-null-budget-sourceL-manysig-m1-r01',
         release_id='cvs_null_budget_sourceL_20261014_r01', request_config='null_sourceL.json')
