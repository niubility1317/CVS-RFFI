import json,torch
from . import design as d
from .checks import checks,smoke_model
from .trace import actual_arguments
from experiments.cvs_neighbor_pair50.predict import source_proof,own_model

def main():
 torch.set_num_threads(2);result=checks()
 for row in d.rows():actual_arguments(source_proof(d.source_config(row)))
 smoke=smoke_model(own_model(d.source_config(d.rows()[0]),'cpu'))
 if (d.ROOT/'preflight.json').exists():raise FileExistsError('Existing source derivative preflight')
 value=dict(status='PASS',checks=result,real_checkpoint_no_query_smoke=smoke,source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0)
 d.write(d.ROOT/'preflight.json',value);print(json.dumps(value))

if __name__=='__main__':main()
