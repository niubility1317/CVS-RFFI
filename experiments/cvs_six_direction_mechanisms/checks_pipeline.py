"""Dispatcher interface/failure fixture; mathematical checks live per mechanism."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from . import run as runner
from . import design as d

def checks():
    roles={}
    for role in d.ROLE_COUNTS:
        cache=dict(x=torch.zeros(2,2,256),rx=torch.tensor([1,3]),day=torch.tensor([1,2]),ids=[role+'0',role+'1'])
        if role!='U_s':cache['y']=torch.tensor([0,1])
        roles[role]=cache
    def model(row,device):return torch.nn.Linear(2,6).eval().requires_grad_(False),dict(source_contract={},source_checkpoint='synthetic-only')
    def context(row,model,roles,device):
        return dict(features={},probabilities={},folds=torch.tensor([0,2]),fold_info={},
                    state_q=torch.zeros(2,12),state_mean=torch.zeros(12),state_scale=torch.ones(12),seed=row['model_seed'])
    def import_module(name,package):
        def diagnose(ctx,path,device):
            path.mkdir(exist_ok=False)
            passed=name!='.updates' or ctx['seed']!=d.SEEDS[1]
            return dict(applicability=dict(passed=passed,criteria='fixed fixture',reason='synthetic'))
        return SimpleNamespace(diagnose=diagnose)
    root=d.ROOT/'local_artifacts';root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='six_pipeline_',dir=root) as folder:
        p=Path(folder)
        with patch.object(runner,'load_roles',return_value=(roles,{})),patch.object(runner,'load_model',side_effect=model),patch.object(runner,'context',side_effect=context),patch.object(runner.importlib,'import_module',side_effect=import_module):
            # The runtime commit file is a release artifact. Supply it only in this fixture.
            original=Path.read_text
            def read(path,*a,**kw):return 'synthetic-only\n' if path.name=='release_commit.txt' else original(path,*a,**kw)
            with patch.object(Path,'read_text',read):done=runner.run(p/'success','cpu',registration=False)
            assert done['rows']==3 and done['direction_cells']==18 and done['identity_updates']==0
            analysis=d.read(p/'success/analysis.json')
            assert analysis['directions']['matching']['eligible_for_preregistered_scratch30']
            assert not analysis['directions']['update']['eligible_for_preregistered_scratch30']
            raw=torch.load(p/'success'/d.rows()[0]['row_id']/'raw_source.pt',weights_only=False)
            assert 'y' not in raw['labels']['U_s'] and not raw['target_access']
            try:runner.run(p/'success','cpu',registration=False)
            except FileExistsError:pass
            else:raise AssertionError('Existing run output overwritten')
            def failed_import(name,package):
                def fail(*args):raise RuntimeError('fixture technical failure')
                return SimpleNamespace(diagnose=fail)
            with patch.object(Path,'read_text',read),patch.object(runner.importlib,'import_module',side_effect=failed_import):
                try:runner.run(p/'failed','cpu',registration=False)
                except RuntimeError:pass
                else:raise AssertionError('Technical failure ignored')
            failed=d.read(p/'failed/failure.json')
            assert failed['status']=='FAILED' and failed['no_automatic_retry'] and not (p/'failed/completion.json').exists()
    return dict(status='PASS',synthetic_only=True,no_actual_checkpoint=True,no_actual_source_IQ=True,
                complete_three_by_six=True,all_seed_condition=True,hidden_U_truth_not_persisted=True,
                existing_root_protected=True,technical_failure_preserved=True,target_access=False)

if __name__=='__main__':print(json.dumps(checks()))
