from pathlib import Path
import json
from . import design as d

def main():
 from .checks import checks
 from .predict import observation_metadata
 from experiments.cvs_e20_foundation.verify import verify
 proof=verify(d.SEEDS[0],build_smoke=True)
 if proof['status']!='VERIFIED' or proof['training_target_access'] or not proof['source_roles']=='EXACT_MATCH':raise ValueError('Source-compatible real checkpoint unavailable')
 ids,metadata=observation_metadata()
 if len(ids)!=d.QUERY_COUNT:raise ValueError('Declared clean capsule metadata differs')
 result=dict(status='PASS',real_checkpoint_no_query_smoke=True,checkpoint=proof['checkpoint'],old_weights_used='smoke only; no training initialization',training_inherits_old_weights=False,source_contract='existing exact physical roles',observation_handles='VALIDATED_ONCE',query_IQ_or_truth_read=False,rows=len(d.rows()),checks=checks())
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Preflight artifact already exists')
 d.write(p,result);print(json.dumps(result))
if __name__=='__main__':main()
