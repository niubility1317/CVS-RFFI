from . import design as d,runtime
from experiments.pamf_fullresolution_restore.training import train as shared_train

def iq_loss(estimate,clean):
    nmse=(estimate-clean).square().mean((1,2))/clean.square().mean((1,2)).clamp_min(1e-4)
    return nmse,dict(iq_nmse=nmse)

def train(args):shared_train(args,configuration=d.configuration,model_runtime=runtime,loss_fn=iq_loss)
