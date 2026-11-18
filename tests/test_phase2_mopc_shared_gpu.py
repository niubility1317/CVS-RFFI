from experiments.pamf_phase2_practical.clean_scheduler import choose


def test_explicit_sharing_admits_one_on_a_busy_gpu_with_sufficient_memory():
    jobs=[{'id':'a'},{'id':'b'}]
    devices={0:{'pids':{999},'blocked':True,'free':7500,'used_by_pid':{999:9000}}}
    assert choose(jobs,{},devices,[0],6000,4,worker_reserve=3000,one_per_snapshot=True)==[]
    chosen=choose(jobs,{},devices,[0],6000,4,worker_reserve=3000,one_per_snapshot=True,allow_shared=True)
    assert [(j['id'],gpu) for j,gpu in chosen]==[('a',0)]


def test_sharing_still_respects_free_memory_and_own_process_limit():
    jobs=[{'id':'a'}]
    devices={0:{'pids':{999},'blocked':True,'free':5999,'used_by_pid':{999:9000}}}
    assert choose(jobs,{},devices,[0],6000,4,worker_reserve=3000,allow_shared=True)==[]
    devices[0]['free']=10000
    active={str(i):{'pid':i,'gpu':0} for i in range(4)}
    assert choose(jobs,active,devices,[0],6000,4,worker_reserve=3000,allow_shared=True)==[]


def test_sharing_counts_all_observed_processes_even_if_method_is_unknown():
    jobs=[{'id':'a'}]
    devices={0:{'pids':{7,8,9},'blocked':False,'free':16000,'used_by_pid':{7:1,8:1,9:1}}}
    active={'ours':{'pid':10,'gpu':0}}
    assert choose(jobs,active,devices,[0],6000,4,worker_reserve=3000,allow_shared=True)==[]
    devices[0]['pids']={7,8}
    assert len(choose(jobs,active,devices,[0],6000,4,worker_reserve=3000,allow_shared=True))==1


def test_sharing_reserves_memory_for_process_with_unknown_or_low_usage():
    jobs=[{'id':'a'}]
    devices={0:{'pids':{999},'blocked':False,'free':7500,'used_by_pid':{999:0}}}
    assert choose(jobs,{},devices,[0],6000,4,worker_reserve=3000,allow_shared=True)==[]
    devices[0]['used_by_pid'][999]=100
    assert choose(jobs,{},devices,[0],6000,4,worker_reserve=3000,allow_shared=True)==[]
    devices[0]['used_by_pid'][999]=4000
    assert len(choose(jobs,{},devices,[0],6000,4,worker_reserve=3000,allow_shared=True))==1
