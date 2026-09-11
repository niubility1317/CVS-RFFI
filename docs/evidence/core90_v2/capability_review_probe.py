"""Read-only CPU reproduction of invalid-current-view handling."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
import torch
from cvsrffi.game_tracking.capability import CapabilityConfig,evaluate_capability_v2
torch.set_num_threads(1)
tx=torch.tensor([0,0,1,1]*2);rx=torch.tensor([0,1,0,1]*2)
z=torch.eye(2)[tx]*3
try:
    result=evaluate_capability_v2(z,tx,rx,z,tx,rx,
        fit_groups=['fit'+str(i) for i in range(8)],monitor_groups=['monitor'+str(i) for i in range(8)],
        monitor_views={'current':z[:-1],'next':z},policy_level=0.,
        config=CapabilityConfig(steps=2),independence_verified=True)
    print(json.dumps({'returned':True,'valid':result['valid'],'reason':result['reason']}))
except RuntimeError as exc:
    print(json.dumps({'returned':False,'exception':type(exc).__name__,'message':str(exc)}))
