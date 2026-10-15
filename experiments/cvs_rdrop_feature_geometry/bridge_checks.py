"""Focused regression for the observed legacy Torch/NumPy bridge failure."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import numpy as np
import torch
from .runtime import feature_array
from .measure import relations

def main():
    torch.set_num_threads(2)
    fit=np.array([(y,rx,day) for y in range(6) for rx in (3,4,6,8) for day in (1,2,3) for _ in range(2)],dtype=np.int64)
    held=np.array([(y,1,day) for y in range(6) for day in (1,2,3) for _ in range(70)],dtype=np.int64)
    labels=[int(v) for v in np.r_[fit[:,0],held[:,0]]]
    z=torch.eye(160,dtype=torch.float32)[labels]
    z[:,159]=.1
    with patch.object(torch.Tensor,'numpy',side_effect=RuntimeError('Numpy is not available')) as bridge:
        array=feature_array(z)
        assert array.shape==(1404,160) and array.dtype==np.float32 and np.all(array[:,159]==np.float32(.1))
        with TemporaryDirectory() as folder:
            path=Path(folder)/'features.npz';np.savez_compressed(path,features=array)
            with np.load(path,allow_pickle=False) as stored:loaded=stored['features'].copy()
        assert np.array_equal(array,loaded)
        result=relations(loaded,fit[:,0],fit[:,1],fit[:,2],held[:,0],held[:,1],held[:,2])
        assert result['groups'][0]['strict_wins']==3528000 and result['groups'][0]['relation_rate']==1
        assert bridge.call_count==0
    print(json.dumps(dict(status='PASS',scope='Only actual feature conversion/save/load/relationship entry',Tensor_numpy_disabled=True,
        Tensor_numpy_calls=0,FP32_roundtrip_exact=True,records=144,aggregate_groups=86,ideal_triplets=3528000,
        real_IQ_reads=0,checkpoint_loads=0,model_forwards=0,optimizer_steps=0,head_fit_updates=0)))

if __name__=='__main__':main()
