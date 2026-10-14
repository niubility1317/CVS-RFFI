import json,torch
from . import design as d
from .checks import checks
from experiments.cvs_source_self_challenge.audit import source_model
from experiments.cvs_signature_stats50.predict import source_proof

def main():
 torch.set_num_threads(2);local=checks();source=source_proof(d.source_config(),design=d.parent)
 checkpoint=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if checkpoint['epoch']!=50 or checkpoint['ema_model'] is None or checkpoint['a1_ema_successful_updates']!=11100:raise ValueError('OwnEMA checkpoint unverified')
 model=source_model().eval().requires_grad_(False);before={k:v.clone() for k,v in model.state_dict().items()}
 with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256))
 if out['id_feat_joint'].shape!=(2,160) or out['tx_logits'].shape!=(2,6) or not torch.isfinite(out['tx_logits']).all() or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Real checkpoint no-query smoke failed')
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Immutable preflight exists')
 value=dict(status='PASS',checks=local,real_checkpoint_no_query_smoke=True,synthetic_IQ_packets=2,source_only=True,query_read=False,head_fit_updates=0,optimizer_updates=0)
 d.write(p,value);print(json.dumps(value))
if __name__=='__main__':main()
