"""One execution entry for deployment, timing and formal prediction."""
import torch
from .program import outputs
from experiments.pamf_r46_cuda_graph.graph import GraphCache, validate_iq


class TensorPipeline:
    def __init__(self,model,reader=None,*,reuse=True,native=False):
        for module in (model,reader):
            if module is not None and (module.training or any(p.requires_grad for p in module.parameters())):
                raise ValueError('Only frozen eval modules may enter inference')
        self.model,self.reader,self.reuse,self.native=model,reader,reuse,native

    def __call__(self,x):
        if self.native:
            iq,variance,_,_=self.model.training_outputs(x)
            valid=torch.stack([torch.isfinite(x).all(),torch.isfinite(iq).all(),torch.isfinite(variance).all(),(variance>0).all()]).all()
        else:
            iq,variance,_,_,valid=outputs(self.model,x,reuse=self.reuse)
        if self.reader is None:
            return iq,valid
        features=self.reader.features(iq)
        logits=self.reader.classify_features(features)
        valid=valid & torch.isfinite(features).all() & torch.isfinite(logits).all()
        return iq,variance.mean(-1),features,logits,valid


class InferenceEngine:
    def __init__(self,model,reader=None,*,reuse=True,graph=True,native=False):
        if native and graph:
            raise ValueError('Native reference is eager only')
        self.pipeline=TensorPipeline(model,reader,reuse=reuse,native=native)
        self.call=GraphCache(self.pipeline) if graph else self.pipeline
        self.graph=graph

    @torch.inference_mode()
    def __call__(self,x):
        validate_iq(x)
        result=self.call(x)
        if not bool(result[-1]):
            raise FloatingPointError('Nonfinite IQ/action/state/F output or nonpositive variance')
        return result[:-1]

    def setup(self):
        return self.call.setup() if self.graph else []
