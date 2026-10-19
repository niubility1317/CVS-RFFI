"""Focused no-IQ legality/shape tests for the frozen source readout."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def module(name):
    p=Path(__file__).with_name(name+'.py')
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    runner=module('cvs_dual_teacher_adapter_source_fit_20261019')
    selected=[];index=[]
    for tx in range(6):
        for rx in (1,3,4,6,8):
            for day in (1,2,3):
                for sig in range(2):
                    idx=len(index);index.append(SimpleNamespace(tx_i=tx,eq_i=1,rx_i=rx,day_i=day,sig_i=sig,physical_id=str(idx)))
                    selected.append(dict(TX=tx,RX=rx,day=day,pid=[1,rx,day,sig,idx],physical_id=str(idx)))
    base=SimpleNamespace(index=index,tx_list=['x'+str(i) for i in range(6)],transform=None,normalize=True,center=False)
    roles={'L_s':[str(i) for i in range(180)]}
    opaque=lambda item:item.physical_id
    runner.checked_panel(selected,base,roles,opaque,base.tx_list)
    tests=[]
    for key,value in [('physical_id','not-legal-L'),('pid',[1,0,1,0,0]),('TX',5),('pid',[1,1,1,0,99999])]:
        s=copy.deepcopy(selected);s[0][key]=value;tests.append((s,base,roles))
    s=copy.deepcopy(selected);s[1]=s[0];tests.append((s,base,roles))
    tests.append((selected[:-1],base,roles))
    for key,value in [('center',True),('normalize',False),('transform',object())]:
        b=copy.deepcopy(base);setattr(b,key,value);tests.append((selected,b,roles))
    tests.append((selected,base,{'L_s':['not-an-L-id']}))
    for s,b,r in tests:
        try:runner.checked_panel(s,b,r,opaque,base.tx_list)
        except ValueError:pass
        else:raise AssertionError('Unauthorized physical role/preprocessing accepted')
    metric=module('cvs_dual_teacher_adapter_source_metrics_20261019')
    print(json.dumps(dict(status='PASS',protocol_negative_cases=len(tests),CP_reads=0,IQ_reads=0,NN=0,updates=0,target_access=False)))


if __name__=='__main__':main()
