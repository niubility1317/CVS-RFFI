from types import SimpleNamespace
import torch
import pytest
from experiments.pamf_phase3_restoration.variance_trial import recovery_gpu_choice
from experiments.pamf_phase3_restoration.multiview_trial import factory,save_recovery


def test_dead_gpu7_cannot_block_or_be_selected_and_busy_cards_are_excluded():
    calls=[]
    def run(command,**kwargs):
        calls.append(command);assert '-i' in command;gpu=int(command[command.index('-i')+1]);assert gpu in (4,6)
        if '--query-gpu=index,uuid,memory.free' in command:return SimpleNamespace(returncode=0,stdout=f'{gpu}, GPU-card{gpu}, 21000\n',stderr='')
        n=2 if gpu==4 else 3
        return SimpleNamespace(returncode=0,stdout=''.join(f'GPU-card{gpu}, {i}\n' for i in range(n)),stderr='')
    choices=recovery_gpu_choice(run)
    assert choices==[dict(GPU=4,CUDA_ID='GPU-card4',free_memory_MB=21000,existing_compute_PIDs=2)]
    assert len(calls)==4


def test_failed_single_card_query_does_not_fall_back_to_unauthorized_gpu():
    def run(command,**kwargs):return SimpleNamespace(returncode=6,stdout='',stderr='device unavailable')
    assert recovery_gpu_choice(run)==[]


def test_epoch_recovery_preserves_model_and_optimizer_without_overwrite(tmp_path):
    torch.set_num_threads(2);g=factory('TF_M20');optimizer=torch.optim.AdamW(g.parameters(),lr=.001)
    for p in g.parameters():p.grad=torch.ones_like(p)
    optimizer.step()
    folder=tmp_path/'recovery';save_recovery(folder,g,optimizer,'TF_M20','a'*40,1,247)
    value=torch.load(folder/'E01.pt',map_location='cpu',weights_only=False)
    assert value['epoch']==1 and value['updates']==247 and value['selection_use'].startswith('technical_recovery_only')
    assert all(torch.equal(v,value['state'][k]) for k,v in g.state_dict().items())
    expected=optimizer.state_dict()
    assert value['optimizer']['param_groups']==expected['param_groups']
    for key,state in expected['state'].items():
        for name,item in state.items():
            torch.testing.assert_close(value['optimizer']['state'][key][name],item,atol=0,rtol=0)
    assert not list(folder.glob('*.partial'))
    with pytest.raises(FileExistsError):save_recovery(folder,g,optimizer,'TF_M20','a'*40,1,247)
    with pytest.raises(ValueError):save_recovery(folder,g,optimizer,'TF_M20','a'*40,2,248)
