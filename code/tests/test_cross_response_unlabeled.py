from types import SimpleNamespace
import torch
from cvsrffi.cross_response.unlabeled import UnlabeledSourceView


class Dataset:
    split_source='ssdg_unlabeled_tx_hidden'
    index=[SimpleNamespace(tx_i=5,rx_i=1,day_i=2,eq_i=0,sig_i=3)]
    def __len__(self):
        return 1
    def __getitem__(self,index):
        return torch.ones(2,16),5,7,{'tx_i':5,'tx':'device_secret','true_tx_i':5,
            'physical_sample_id':(5,1,2,0,3),'rx_i':1,'day_i':2,'eq_i':0,'sig_i':3,
            'base_index':123,'split_source':self.split_source}


def test_unlabeled_payload_hides_truth_but_retains_legal_domain_and_identity_for_contract():
    ds=Dataset()
    view=UnlabeledSourceView(ds)
    assert view.index is ds.index and view.split_source==ds.split_source
    x,y,d,meta=view[0]
    assert y==-1 and d==7 and torch.equal(x,ds[0][0])
    assert not any(key in meta for key in ('tx_i','tx','true_tx_i','physical_sample_id'))
    assert meta['base_index']==123 and meta['sig_i']==3 and meta['tx_label_visible'] is False
    batch=next(iter(torch.utils.data.DataLoader(view,batch_size=1)))
    assert batch[1].tolist()==[-1] and batch[3]['tx_label_visible'].tolist()==[False]
