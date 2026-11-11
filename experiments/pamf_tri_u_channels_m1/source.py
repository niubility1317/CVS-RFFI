from contextlib import contextmanager
from experiments.pamf_branch_training.source import checked_e20_parent
from experiments.pamf_paper_completion import model,plan as paper
from experiments.pamf_repl_completion import objectives
from experiments.pamf_tri_u_phase import objective,source as phase
from experiments.pamf_tri_u_channels.transform import channel_view
from .plan import ARMS,validate_candidate


def parent(c):
    validate_candidate(c);result=checked_e20_parent(c);cfg=result['config'];row=c['paper_row']
    if (cfg.get('run_id')!=paper.RUN or cfg.get('row_id')!=row['row_id'] or cfg.get('paper_row')!=row
        or cfg.get('protocol_id')!=row['protocol_id'] or cfg.get('source_cache')!=c['source_cache']
        or cfg.get('source_rxs')!=c['source_rxs'] or cfg.get('source_days')!=c['source_days']):
        raise ValueError('Original single-source-RX same-seed E20 source or ancestry differs')
    return result


@contextmanager
def installed(c,holder):
    validate_candidate(c);arm=c['algorithm_id']
    saved=paper.validate_config,objectives.objective,objective.rotate_iq
    paper.validate_config=validate_candidate;objectives.objective=objective.GuardedPhaseU
    objective.rotate_iq=lambda x,angle:channel_view(x,angle,**ARMS[arm])
    try:
        with model.installed(c,holder,ssl_enabled=True):yield
    finally:
        paper.validate_config,objectives.objective,objective.rotate_iq=saved


def train(c):
    old=phase.parent,phase.installed;phase.parent,phase.installed=parent,installed
    try:return phase.train(c)
    finally:phase.parent,phase.installed=old
