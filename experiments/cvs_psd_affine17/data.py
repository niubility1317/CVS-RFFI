"""Reuse only the existing approved fit-L aggregate;no IQ container or weights."""
import json
import numpy as np
from . import design as d

def metadata(config,summary):
 expected=dict(run_id=d.config()['source_origin_run'],source_RX=[3,4,6,8],source_TX=list(range(6)),source_days=[1,2,3],L_packets=5040,packets_per_cell=70,source_only=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,model_forwards=0,checkpoint_loads=0,optimizer_updates=0,new_models=0)
 if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Existing source aggregate roles/schema differ')
 if config['observation']['windows']!=['rectangular','hann'] or config['observation']['FFT_bins']!=256:raise ValueError('Original full-bin axis definition differs')
 if summary.get('physical_packet_counts')!={'L':5040,'U':0,'V':0,'held_RX1':0,'target':0} or summary.get('source_cells')!=72:raise ValueError('Original full fit-L cells differ')
 return dict(source_origin_run=expected['run_id'],physical_packet_counts=summary['physical_packet_counts'],source_cells=72,axis_definition='window,TX,day,RX,shiftedbin;original producer loops checked',reuse='Existing VALIDATED_ONCE fit-L aggregate;not new data validation or IQ access')
def load():
 config=json.loads((d.SOURCE/'resolved_config.json').read_text(encoding='utf8'));summary=json.loads((d.SOURCE/'summary.json').read_text(encoding='utf8'));evidence=metadata(config,summary)
 with np.load(d.SOURCE/'source_line_statistics.npz',allow_pickle=False) as z:psd=np.asarray(z['cell_PSD'],dtype=np.float64)
 return psd,evidence
