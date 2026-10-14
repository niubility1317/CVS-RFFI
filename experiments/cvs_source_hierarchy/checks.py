import json,torch
from .algebra import checks as algebra,components
from .probe import statistics,smoke
def checks():
 torch.set_num_threads(2);a=algebra()
 from experiments.cvs_moment_area50.model import build
 from . import design as d
 model=build(d.source_config(),'cpu').eval();s=smoke(model)
 torch.manual_seed(53);u=torch.rand(720,32,6).double()*.8-.4;c=components(u);rx=torch.tensor([r for r in d.RX for _ in range(144)]);day=torch.tensor([r for r in d.DAYS for _ in range(240)]);records=statistics(c,rx,day);assert len(records)==135
 time=[r for r in records if r['component']=='time_increment1' and r['dimension']=='overall'];by={r['variant']:r for r in time}
 assert abs(by['missing_bridges']['error_RMS']-3/31)<1e-14 and abs(by['reset_clock']['error_RMS']-(3+3/31))<1e-14 and by['omitted_cross_area']['error_RMS']<1e-14
 from .publish import configure,INSPECT
 from experiments.cvs_report_ssl import publish as p
 configure();compile(p.REMOTE,'REMOTE','exec');compile(INSPECT,'INSPECT','exec');assert 'experiments/cvs_source_dropout' in p.PATHS and 'experiments/cvs_source_hierarchy' in p.PATHS
 return dict(status='PASS',algebra=a,native_no_query_smoke=s,synthetic_stats_count135=True,clock_error_controls_checked=True,release_deps_compile=True,source_or_target_IQ_read=False)
if __name__=='__main__':print(json.dumps(checks()))
