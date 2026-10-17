"""Draw the existing receiver prior once per label-independent virtual session."""
import re
from . import design as d
def draw(cfg,ch,session,epoch):
 if isinstance(epoch,bool) or not isinstance(epoch,int) or epoch not in range(21,31):raise ValueError('Cumulative epoch21..30 only')
 if not isinstance(session,str) or re.fullmatch(r'rx(1|3|4|6|8):day(1|2|3):eq1',session) is None:raise ValueError('Original RX/day/eq session only,no TX key')
 if (cfg.iq_amp_max_db,cfg.iq_phase_max_deg,cfg.receiver_cfo_std_hz,cfg.iq_estimation_amp_std_db,cfg.iq_estimation_phase_std_deg,cfg.iq_compensation_enabled,cfg.equalization_enabled,cfg.processing_route,cfg.mode,cfg.reference_snr_db)!=(.2,1.,200.,.05,.2,True,False,'residual','post_sync',30.):raise ValueError('Original receiver and complete residual prior required')
 return ch.receiver_for_session(cfg,d.POPULATION_SEED,f'virtual/e{epoch}/{session}')
