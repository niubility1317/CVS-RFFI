from . import design as d,runtime
from experiments.pamf_fullresolution_restore.training import train as original_train
from experiments.pamf_joint_coarse_restore.training import iq_loss


def train(args):
    return original_train(args,configuration=d.configuration,model_runtime=runtime,loss_fn=iq_loss)
