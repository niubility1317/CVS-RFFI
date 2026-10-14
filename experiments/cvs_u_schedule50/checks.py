from types import SimpleNamespace
from . import design as d
def run():
 from experiments.cvs_phase1_overlay.model import native_modules
 n=native_modules()
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));a=d.make_args(c,'cpu');phases=[n._source_epoch_phase(e,a) for e in range(1,51)]
  assert [e for e,p in enumerate(phases,1) if p=='pseudo']==d.CALENDARS[arm] and phases.count('label')==33 and phases.count('pseudo')==17
  assert len(c['calendar']['optimizer_probe_global_steps'])==4 and len(set(c['calendar']['optimizer_probe_global_steps']))==4
 a.source_unlabeled_epochs=None;assert [e for e in range(1,51) if n._source_epoch_phase(e,a)=='pseudo']==d.CALENDARS['TAIL']
 count=0
 for bad in ([],[17]*17,list(range(1,19)),list(range(34,50))+[51],list(range(34,50))+[49.0],list(reversed(range(34,51)))):
  a.source_unlabeled_epochs=bad
  try:n._source_epoch_phase(1,a)
  except ValueError:count+=1
  else:raise AssertionError('Invalid source schedule accepted')
 return dict(status='PASS',equal17U_33label_calendar=True,legacy_absent_calendar_exact=True,invalid_calendar_negative_cases=count)
