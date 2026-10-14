"""Post-completion recount of fixed remaining rows; read-only, no training."""
import json
from . import design as d
from experiments.cvs_identity_mixstyle50.audit import run as shared_run
def run():return shared_run(False,design=d)
if __name__=='__main__':print(json.dumps(run()))
