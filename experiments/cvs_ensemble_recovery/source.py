from pathlib import Path
from . import design as d
from experiments.cvs_ensemble_final30 import design as original
from experiments.cvs_ensemble_final30.predict import source_proof as original_proof
from experiments.cvs_ensemble_final30.contracts import verified_component_contract
from experiments.cvs_phase1_stack.recover import verify_contract

def input_proof(c):
 d.validate(c);old=c['source_config'];original.validate(old);source=original_proof(old,repair_reference_copy=True)
 row=next(r for r in original.rows() if r['row_id']==old['row_id'])
 contract=verified_component_contract([d.read(Path(original.member_config(row,i)['output'])/'source_contract.json') for i in (1,2)],d.read(d.SOURCE))
 verify_contract(contract,d.read(d.SOURCE));return source,contract
def prepare(rowid):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));source,contract=input_proof(c);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
 d.write(out/'source_contract.json',contract)
 d.write(out/'identity.json',dict(config=c,original_source=str(source),original_bundle=str(source/'identity.pth'),model_updates=0,source_members_final_epoch=30,formula=c['ensemble']['formula'],source_roles='EXACT_MATCH',target_access=False))
 d.write(out/'completion.json',dict(status='SOURCE_REUSED_FROZEN',config=c,source_roles='EXACT_MATCH',new_model_updates=0,target_access=False))
 d.write(out/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,checkpoint=str(source/'identity.pth'),selected_by='originalfixedfinal30sourcequalifiedrecipe',target_access=False))
 return out
