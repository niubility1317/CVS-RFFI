"""Seven-view evaluation of the full-rollout-trained model."""
from . import historical_eval as core
from . import rollout_train
core.candidate=rollout_train

if __name__=='__main__':core.main()
