from . import design as d,runtime
from experiments.pamf_fullresolution_restore.training import train as shared_train
from experiments.pamf_iqonly_restore.training import iq_loss


def train(args):shared_train(args,configuration=d.configuration,model_runtime=runtime,loss_fn=iq_loss)
