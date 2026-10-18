"""Check author four-fold source ordering and target exclusion without real IQ."""
import copy
import json
import pathlib
import tempfile
from unittest import mock
import numpy as np
import pipeline as p

conf=p.read_json(pathlib.Path(__file__).with_name('candidate.json'))
seen=[]
with tempfile.TemporaryDirectory(prefix='ibrff-folds-') as tmp:
    roles=pathlib.Path(tmp)/'roles.json'
    original=p.contract(conf)
    p.write_json(roles,{'rx_author_order':conf['rx_author_order'],
                      'source':original['source_rx'],'target':original['target_rx'],
                      'asknet_wavemlp':{'packet_rows':list(range(100))}})
    for fold in range(4):
        c=copy.deepcopy(conf);c['round']=fold
        ct=p.contract(c)
        expected_target=conf['rx_author_order'][3*fold:3*fold+3]
        assert ct['target_rx']==expected_target
        assert ct['source_rx']==[rx for rx in conf['rx_author_order'] if rx not in expected_target]
        assert not set(ct['source_rx']) & set(ct['target_rx'])
        seen.extend(ct['target_rx']);calls=[]
        def source_spy(data,rx,day):
            assert rx in ct['source_rx'] and rx not in ct['target_rx'] and day in [1,2]
            calls.append((rx,day))
            return np.zeros((600,2,256),np.float32),np.repeat(np.arange(6),100)
        with mock.patch.object(p,'read_block',side_effect=source_spy):
            train,val=p.source_data({'physical_roles':str(roles),'data_root':'unused'},c)
        assert calls==[(rx,day) for rx in ct['source_rx'] for day in [1,2]]
        assert len(train[0])==7560 and len(val[0])==3240
    assert seen==conf['rx_author_order']
    for invalid in [-1,4,True,1.5]:
        c=copy.deepcopy(conf);c['round']=invalid
        try:p.contract(c)
        except ValueError:pass
        else:raise AssertionError('Invalid fold accepted')
print(json.dumps({'status':'PASS','folds':4,'each_target_rx_occurs_once':True,
                  'source_paths_target_excluded':True,'author_rx_day_order_preserved':True,
                  'round0_contract_unchanged':original==p.contract(conf)}))
