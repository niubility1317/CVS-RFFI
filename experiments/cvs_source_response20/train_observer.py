from collections import Counter
from contextlib import contextmanager
from . import train_design as d
from experiments.cvs_mixed_iq20.train_observer import Observer as OriginalObserver


class Observer(OriginalObserver):
    def __init__(self,c):
        super().__init__(c);self.response_coverage=Counter();self.sessions={};self.response_packets=0

    @contextmanager
    def native_scope(self,n):
        with super().native_scope(n):
            previous=n._detach_log_mapping
            def logged(value):
                result=previous(value)
                if 'train/loss' in result:
                    records=self.m._response_records;self.m._response_records=[]
                    if len(records)!=128:raise ValueError('Actual G metadata count differs')
                    for v in records:
                        key=(v['epoch'],v['RX'],v['day'],v['eq'],v['view']);parameters=(v['bank_index'],v['strength'])
                        if key in self.sessions and self.sessions[key]!=parameters:raise ValueError('G session varies with TX or packet')
                        self.sessions[key]=parameters;self.response_coverage[(*key,v['TX'])]+=1
                    self.response_packets+=len(records)
                return result
            n._detach_log_mapping=logged
            try:yield
            finally:n._detach_log_mapping=previous

    def finish(self):
        super().finish()
        expected={(e,rx,day,1,0) for e in range(1,21) for rx in (1,3,4,6,8) for day in (1,2,3)}
        if self.response_packets!=568320 or set(self.sessions)!=expected or set(self.response_coverage)!={(*v,tx) for v in expected for tx in range(6)}:
            raise ValueError('G epoch x RX/day session x TX support incomplete')
        d.write(self.out/'response_coverage_complete.json',dict(status='VERIFIED',packets=self.response_packets,
            sessions=len(self.sessions),all_TX_each_session=True,view=0,kind=self.c['arm'],TX_used_for_parameters=False,
            records=[dict(epoch=k[0],RX=k[1],day=k[2],eq=k[3],view=k[4],TX=k[5],count=v,
                bank_index=self.sessions[k[:5]][0],strength=self.sessions[k[:5]][1]) for k,v in sorted(self.response_coverage.items())]))
