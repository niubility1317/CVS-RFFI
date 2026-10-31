"""Seven-view evaluation of the full-G10 rollout-trained model."""
from . import historical_eval as core
from . import fullg10_rollout_train
core.candidate=fullg10_rollout_train

if __name__=='__main__':core.main()
