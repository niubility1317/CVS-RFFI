import torch
from experiments.phase1_dura_orthogonal.benchmark_shared_direct_gates import predict, compare


class Reader:
    def features(self,x):return x
    def classify_features(self,x):return x.mean(-1)


def test_shares_context_and_direct_without_output_changes():
    calls={'context':0,'direct':0}
    x=torch.ones(2,2,4)
    def context(v):
        calls['context']+=1
        return v*.5,v*.25,torch.ones(len(v))
    def direct(v):
        calls['direct']+=1
        return v*.75
    gates=(lambda x,a,d,p,v:a+.1*(p-a),lambda x,a,d,p,v:a+.8*(p-a))
    baseline=predict(x,context,direct,Reader(),gates,False)
    assert calls=={'context':2,'direct':2}
    calls.update(context=0,direct=0)
    shared=predict(x,context,direct,Reader(),gates,True)
    assert calls=={'context':1,'direct':1}
    compare(baseline,shared)
