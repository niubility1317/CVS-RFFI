
import json,sys
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_lora_main.private_dose_source import summarize_telemetry
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.complex_vector_source import load_reader
from experiments.pamf_lora_main.channel_blind_source import setup
rt=json.loads(Path('runtime.json').read_text())
activate(rt)
_,contract=setup(rt,'cpu',smoke=True)
reader=load_reader(Path(rt['run_root']),contract,'cpu')
converted=torch.ones((2,2,4,4)).numpy()
empty=np.empty((0,2,4,4))
print(json.dumps(dict(numpy=np.__version__, torch=torch.__version__,
   converted_type_matches_np=type(converted) is np.ndarray, empty_type_matches_np=type(empty) is np.ndarray,
   types_match=type(converted) is type(empty), current_module=('numpy' in sys.modules and np is sys.modules['numpy']))),flush=True)
for label,rows in (('torch-only',[(converted,0.)]),('mixed',[(empty,0.),(converted,0.)])):
    try:
        v=summarize_telemetry(rows)
    except TypeError as e:
        print(label,type(e).__name__,str(e),flush=True)
    else:
        print(label,'PASS',v['active'],flush=True)
