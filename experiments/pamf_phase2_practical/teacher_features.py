"""Frozen default-head probabilities; no new classification head is fitted."""
import numpy as np
from .milkr.orbit_cca import OrbitCCA
from .milkr.source_bridge import BLOCKS

PHASE_KEY='teacher_phase_logits'


def probabilities(logits,old_columns=None):
    value=np.asarray(logits,np.float64)
    if value.ndim!=3 or value.shape[1:]!=(4,6) or not np.isfinite(value).all():raise ValueError('Expected actual four-phase six-class default logits')
    columns=list(range(6)) if old_columns is None else list(old_columns)
    if len(columns)<2 or len(set(columns))!=len(columns) or any(c not in range(6) for c in columns):raise ValueError('Invalid old-head columns')
    # Remove held-new head columns BEFORE softmax in source old4/new2 proxies.
    selected=value[:,:,columns];selected=np.exp(selected-selected.max(2,keepdims=True))
    return (selected/selected.sum(2,keepdims=True)).mean(1)


def ordinary(features):
    if set(features)!=set(BLOCKS[:5])|{'sig_views',PHASE_KEY}:raise ValueError('Exact teacher feature payload required')
    return {key:value for key,value in features.items() if key!=PHASE_KEY}


def extract(reader,raw,device,engine=None,batch=128):
    from .phase_orbit import extract as orbit_extract
    features,logits=orbit_extract(reader,raw,device,engine,batch,return_phase_logits=True)
    return dict(features,teacher_phase_logits=logits),logits[:,0,:]


def extract_time(reader,raw,device,engine=None,batch=128):
    from .phase_orbit import extract as orbit_extract,spacetime_moments
    features,logits=orbit_extract(reader,raw,device,engine,batch,aggregate=spacetime_moments,return_phase_logits=True,time_shifts=(0,64,128,192))
    # The neural-head prior keeps its original unshifted C4 probability function.
    return dict(features,teacher_phase_logits=logits),logits[:,0,:]


def extract_phase8(reader,raw,device,engine=None,batch=128):
    from .phase_orbit import extract as orbit_extract,eighth_moments
    features,logits=orbit_extract(reader,raw,device,engine,batch,aggregate=eighth_moments,return_phase_logits=True,phase_eighths=True)
    # RF uses C8, while the function prior keeps the original four even phases.
    return dict(features,teacher_phase_logits=logits),logits[:,0,:]


def extract_orbit_ensemble(reader,raw,device,engine=None,batch=128):
    from .phase_orbit import extract as orbit_extract,ensemble_moments
    features,logits=orbit_extract(reader,raw,device,engine,batch,aggregate=ensemble_moments,return_phase_logits=True,phase_eighths=True)
    return dict(features,teacher_phase_logits=logits),logits[:,0,:]


class TeacherBridge:
    encoder_type=OrbitCCA
    def __init__(self,cca):self.cca=cca;self.rank=64

    @classmethod
    def fit(cls,raw,restored):return cls(cls.encoder_type.fit(ordinary(raw),ordinary(restored)))

    def transform(self,features,*,restored,old_columns=None):
        out=self.cca.transform(ordinary(features),restored=restored)
        out['teacher_probs']=probabilities(features[PHASE_KEY],old_columns)
        return out

    def save(self,path):self.cca.save(path)

    @classmethod
    def load(cls,path):return cls(cls.encoder_type.load(path))


class NativeSkipBridge(TeacherBridge):
    from .milkr.native_skip import NativeSkip as encoder_type


class ResidualSkipBridge(TeacherBridge):
    from .milkr.residual_skip import ResidualSkip as encoder_type


class EpisodicMetricBridge(TeacherBridge):
    from .milkr.episodic_metric import EpisodicMetric as encoder_type

    @classmethod
    def fit(cls,raw,restored,**source_metadata):
        return cls(cls.encoder_type.fit(ordinary(raw),ordinary(restored),**source_metadata))


class OrbitEnsembleBridge:
    encoder_type=OrbitCCA
    schema='source_C4_C8_CCA_v1'
    def __init__(self,encoders):
        if len(encoders)!=2 or not all(isinstance(value,OrbitCCA) for value in encoders):raise ValueError('Two ordinary source CCA encoders required')
        self.encoders=tuple(encoders);self.rank=64

    @staticmethod
    def split(features):
        names=tuple(BLOCKS[:5])+('sig_views',)
        if set(features)!=set(names)|{'C4_'+key for key in names}|{PHASE_KEY}:raise ValueError('Exact dual-orbit source payload required')
        return ({key:features['C4_'+key] for key in names},{key:features[key] for key in names})

    @classmethod
    def fit(cls,raw,restored):
        a=cls.split(raw);b=cls.split(restored)
        return cls([cls.encoder_type.fit(x,y) for x,y in zip(a,b)])

    def transform(self,features,*,restored,old_columns=None):
        values=[encoder.transform(part,restored=restored) for encoder,part in zip(self.encoders,self.split(features))]
        out=dict(values[1]);out.update({'C4_'+key:value for key,value in values[0].items()});out['teacher_probs']=probabilities(features[PHASE_KEY],old_columns)
        return out

    def save(self,path):
        from pathlib import Path
        if Path(path).exists():raise FileExistsError('Frozen dual-orbit encoder exists')
        payload=dict(schema=np.asarray(self.schema),rank=np.asarray(64,np.int64))
        for orbit,encoder in zip((4,8),self.encoders):
            for name,p in zip(BLOCKS,encoder.parameters):
                for key,a in zip(('mean_raw','mean_restored','raw_projection','restored_projection'),p):payload[f'C{orbit}_'+name+'_'+key]=a
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path):
        encoders=[]
        with np.load(path,allow_pickle=False) as z:
            if str(z['schema'].item())!=cls.schema or int(z['rank'].item())!=64:raise ValueError('Wrong dual-orbit source checkpoint')
            for orbit in (4,8):
                parameters=[tuple(np.array(z[f'C{orbit}_'+name+'_'+key],dtype=np.float64,copy=True) for key in ('mean_raw','mean_restored','raw_projection','restored_projection')) for name in BLOCKS]
                encoders.append(cls.encoder_type(parameters))
        return cls(encoders)


class CorrelationEnsembleBridge(OrbitEnsembleBridge):
    from .milkr.correlation_cca import CorrelationCCA as encoder_type
    schema='source_C4_C8_correlation_weighted_CCA_v1'


class CrossRXResidualBridge(CorrelationEnsembleBridge):
    from .milkr.crossrx_residual import CrossRXResidualCCA as encoder_type
    schema='source_C4_C8_crossrx_residual_v1'

    @classmethod
    def fit(cls,raw,restored,*,labels,receivers,physical_ids,seed,**training_options):
        a=cls.split(raw);b=cls.split(restored)
        return cls([cls.encoder_type.fit(x,y,labels=labels,receivers=receivers,physical_ids=physical_ids,seed=seed+orbit*100003,**training_options) for orbit,x,y in zip((4,8),a,b)])

    def save(self,path):
        from pathlib import Path
        if Path(path).exists():raise FileExistsError('Frozen cross-RX source encoder exists')
        payload=dict(schema=np.asarray(self.schema),rank=np.asarray(64,np.int64),bound=np.asarray(.25),canonical_power=np.asarray(.5))
        for orbit,encoder in zip((4,8),self.encoders):
            for name,p in zip(BLOCKS,encoder.parameters):
                for key,a in zip(('mean_raw','mean_restored','raw_projection','restored_projection'),p):payload[f'C{orbit}_'+name+'_'+key]=a
            for path_index,path_parameters in enumerate(encoder.neural):
                for block_index,p in enumerate(path_parameters):
                    for key,a in zip(('w1','b1','w2','b2'),p):payload[f'C{orbit}_network_{path_index}_{block_index}_{key}']=a
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path):
        encoders=[]
        with np.load(path,allow_pickle=False) as z:
            if str(z['schema'].item())!=cls.schema or int(z['rank'].item())!=64 or float(z['bound'].item())!=.25 or float(z['canonical_power'].item())!=.5:raise ValueError('Wrong source cross-RX residual checkpoint')
            for orbit in (4,8):
                parameters=[tuple(np.array(z[f'C{orbit}_'+name+'_'+key],np.float64,copy=True) for key in ('mean_raw','mean_restored','raw_projection','restored_projection')) for name in BLOCKS]
                neural=tuple(tuple(tuple(np.array(z[f'C{orbit}_network_{path_index}_{block_index}_{key}'],np.float32,copy=True) for key in ('w1','b1','w2','b2')) for block_index in range(8)) for path_index in (0,1))
                encoders.append(cls.encoder_type(parameters,neural))
        return cls(encoders)


def bridge_type(method):
    if method=='crossrx_residual_ensemble':return CrossRXResidualBridge
    if method=='corr_orbit_ensemble':return CorrelationEnsembleBridge
    if method=='native_skip':return NativeSkipBridge
    if method=='residual_skip':return ResidualSkipBridge
    if method=='episodic_metric':return EpisodicMetricBridge
    if method in ('orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):return OrbitEnsembleBridge
    return TeacherBridge
