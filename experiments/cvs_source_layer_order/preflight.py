import json,torch
from . import design as d
from .checks import checks
from .probe import model,smoke
from experiments.cvs_signature_stats50.predict import source_proof
from experiments.cvs_trusted_weight50 import design as pd

def main():
 torch.set_num_threads(2);p=source_proof(d.source_config(),design=pd);value=dict(status='PASS',checks=checks(),real_checkpoint_no_query_smoke=smoke(model()),source_checkpoint=str(p/'final_ssdg.pth'),query_read=False,U_truth_read=False,optimizer_updates=0,head_fit_updates=0)
 if (d.ROOT/'preflight.json').exists():raise FileExistsError('Existing preflight, reconcile')
 d.write(d.ROOT/'preflight.json',value);print(json.dumps(value))
if __name__=='__main__':main()
