import json,torch
from . import design as d
from .probe import smoke
from experiments.cvs_moment_area50.predict import source_proof,own_model

def main():
 torch.set_num_threads(2);source=source_proof(d.source_config());result=smoke(own_model(d.source_config(),'cpu'))
 if (d.ROOT/'preflight.json').exists():raise FileExistsError('Existing preflight; reconcile')
 value=dict(status='PASS',real_checkpoint_no_query_smoke=result,checkpoint=str(source/'final_ssdg.pth'),source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0)
 d.write(d.ROOT/'preflight.json',value);print(json.dumps(value))
if __name__=='__main__':main()
