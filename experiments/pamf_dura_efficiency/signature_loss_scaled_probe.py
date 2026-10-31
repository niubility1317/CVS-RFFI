"""Separate source hypothesis: shrink signature IQ steps near small loss."""
import argparse
from pathlib import Path
import torch
from .signature_probe import run
from .signature_iq import loss_scaled_signature_steps

RUN='20261031-diagnostic-DURA-source-signatureLossStep-s2026092701-r01'
RULE=('New independent source algorithm after fixed-normalized-step clean overshoot; both prior PA and fixed-step FAIL unchanged. '
 'Same defaultDURA/source512128L384U/caches01and20/pairedclean weightedTripleSIG. '
 'Fourfixedfinal steps: delta=-(L/||gradL||^2)gradL;gradientenergyfloor1e-20;eachstepcap0.25percent nativeIQ RMS;totalproject1percent. '
 'No backtracking/classloss/cleanlabelroute;IQ_oracle closestpairedclean in SAMEball. '
 'UNCHANGED BOTHviewcriteria:signature4 cleanFchoicegain>=4vsnative and>=2vsIQoracle;all5sourceRXgain>=0;eachnonclean scene gain>=-1;clean gain>=0; '
 'mean signature relativeMSE <=0.9native;meanIQNMSE<=1.01native. No stepsize/iterations/reference/parent/threshold search. '
 'PASS supports separate scratchE30 registration,not automatictraining/targetclaim. Sourceclean unavailabledeployment;notphysicalceiling/trueTX.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2)
    run(args,run_id=RUN,rule=RULE,iq_steps=loss_scaled_signature_steps)


if __name__=='__main__':main()
