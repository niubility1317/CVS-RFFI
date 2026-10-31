"""Fixed SOURCE response+SIG counterfactual; no G or new model training."""
from .joint_branch_probe import main

RUN='20261031-diagnostic-DURA-source-response-SIG-s2026092701-r01'
RULE=('Fixed defaultDURA/source512128L384U/caches01and20;cachedactualweightedfourFbranches. '
 'Cases native/responseclean/SIGclean/responseplusSIGclean/allclean,otheridentity-PAunchanged,originalfrozenF50/defaulthead. '
 'PrimaryresponseplusSIGmustBOTHviews:gainvsnative>=6/512,gainvsSIG>=2/512,all5RXgainvsnative>=0,eachRXgainvsSIG>=-1,atleast4of5RXgainvsSIG>=0; '
 'eachnonclean scene gainvsnative>=-1,clean gainvsnative>=0. No branch/parent/threshold search. '
 'Passsupportsseparatelearninghypothesisonly/notautomaticGtraining. OldPA/PAplusSIG/sourceIQFAILspreserved. '
 'Thislatentpairedsourcecleanoracle isnotrealizableIQ/deployablemethod/trueTXaccuracy/physicalrecoveryceiling.')

if __name__=='__main__':
    main(run_id=RUN,auxiliary='response',rule=RULE)
