# Reuse the exact existing reduced architectures, including their shared-weight topology.
from experiments.pamf_r38_successor.models import CompactSuccessor,compact_from_parent
from experiments.pamf_iq_joint_compact.models import objective,convolution_linear_macs
