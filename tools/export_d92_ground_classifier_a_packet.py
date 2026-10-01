"""Export only the explicitly bound source-only ground CosFace classifier head.

No encoder is constructed or run. No samples, feature cache, support/query,
checkpoint metrics, or prototype are inspected. Provenance is the existing
owner-supplied source-only conclusion, not a new approval or verification chain.
"""
import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib
from importlib.machinery import PathFinder
import json
from pathlib import Path
import sys
import time
from types import ModuleType

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.d92_ground_classifier_a import (
    GroundClassifierA,GroundFeatureContract,GroundHeadMetadata,WEIGHT_KEY,
    FEATURE_DIM,GROUND_CLASS_COUNT,SOURCE_ONLY_VERDICT,
)

SCHEMA='d92_ground_classifier_a_packet_v1'
STATUS='GROUND_CLASSIFIER_A_PACKET_EXPORTED'
WEIGHT_FILE='head_weight.float32.bin'
METADATA_FILE='metadata.json'
COMPLETE_FILE='complete.json'
WEIGHT_BYTES=GROUND_CLASS_COUNT*FEATURE_DIM*4
NATIVE_TYPES={'baseline_origin_sat_view':'SatViewStage','cvsrffi.muse_ssdg':'RC4Calibration'}
BINDING_FIELDS={'checkpoint_sha256','checkpoint_weight_key','ordered_classes','scale','norm_eps',
    'feature_contract','logit_corrections','class_row_order_source','factory_scale_source','corrections_source'}


def require(condition,message):
    if not condition:raise ValueError(message)


def encode_json(value):return (json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)+'\n').encode('utf-8')


def bound_metadata(binding,provenance):
    """Consume declarations without inferring class order, scale or corrections."""
    require(isinstance(binding,Mapping) and set(binding)==BINDING_FIELDS,'Explicit complete ground-head binding required')
    require(isinstance(provenance,Mapping),'Existing source-only provenance metadata required')
    for name in ('class_row_order_source','factory_scale_source','corrections_source'):
        value=binding[name]
        require(isinstance(value,str) and value.strip() and value.strip().lower() not in ('unknown','n/a','none','unverified'),
            'Actual binding source is missing: '+name)
    digest=binding['checkpoint_sha256']
    require(provenance.get('checkpoint_sha256')==digest,'Checkpoint/provenance identity mismatch')
    require(provenance.get('verdict')==SOURCE_ONLY_VERDICT and provenance.get('target_access_before_freeze') is False
        and isinstance(provenance.get('checkpoint_inheritance'),(list,tuple)) and not provenance['checkpoint_inheritance'],
        'Existing matched source-only scratch verdict required')
    contract=GroundFeatureContract(**binding['feature_contract'])
    metadata=GroundHeadMetadata(checkpoint_sha256=digest,checkpoint_weight_key=binding['checkpoint_weight_key'],
        ordered_classes=binding['ordered_classes'],scale=binding['scale'],norm_eps=binding['norm_eps'],feature_contract=contract,
        source_only_verdict=provenance['verdict'],target_access_before_freeze=provenance['target_access_before_freeze'],
        checkpoint_inheritance=provenance['checkpoint_inheritance'],logit_corrections=binding['logit_corrections'])
    source_classes=provenance.get('classes')
    require(isinstance(source_classes,(list,tuple)) and len(source_classes)==GROUND_CLASS_COUNT
        and all(isinstance(v,str) and v for v in source_classes) and len(set(source_classes))==GROUND_CLASS_COUNT
        and set(source_classes)==set(metadata.ordered_classes),'Classifier classes differ from verified source registry')
    # A source registry is a set constraint. Its order must not replace the
    # explicitly supplied, independently established native classifier row order.
    retained={k:provenance[k] for k in ('verdict','checkpoint_sha256','target_access_before_freeze',
        'checkpoint_inheritance','classes','model_seed','checkpoint_epoch','source_role_comparison') if k in provenance}
    encode_json(retained)
    return metadata,retained


def extract_exact_weight(payload):
    """Match the native Phase1 'model' state, with one module. prefix at most."""
    require(isinstance(payload,Mapping) and isinstance(payload.get('model'),Mapping),
        "Native checkpoint must contain the explicit 'model' state mapping")
    state=payload['model'];normalized={};original={}
    for key,value in state.items():
        require(isinstance(key,str),'Checkpoint state keys must be strings')
        canonical=key[7:] if key.startswith('module.') else key
        require(canonical not in normalized,'Ambiguous module. prefix normalization: '+canonical)
        normalized[canonical]=value;original[canonical]=key
    require(WEIGHT_KEY in normalized,'Exact native CosFace head weight is missing')
    weight=normalized[WEIGHT_KEY]
    require(isinstance(weight,torch.Tensor) and weight.dtype==torch.float32 and weight.layout==torch.strided
        and tuple(weight.shape)==(GROUND_CLASS_COUNT,FEATURE_DIM),'Original head must be float32[6,160]')
    require(bool(torch.isfinite(weight).all()),'Original head has nonfinite values')
    return weight.detach().cpu().contiguous().clone(),original[WEIGHT_KEY]


def checkpoint_digest(stream):
    digest=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


@contextmanager
def native_checkpoint_types(native_code):
    """Resolve original producer types only during trusted checkpoint loading.

    This is not a sandbox for arbitrary Python or pickle. The caller has already
    bound the original source-only checkpoint and its original training release.
    No substitute classes or encoder factory calls are made here.
    """
    if native_code is None:
        yield None
        return
    declared=Path(native_code)
    require(declared.is_absolute() and declared.is_dir(),'Explicit existing native code directory required')
    native=declared.resolve();package=importlib.import_module('cvsrffi')
    origins={name:native/('cvsrffi/muse_ssdg.py' if '.' in name else name+'.py') for name in NATIVE_TYPES}
    for name,path in origins.items():
        require(path.is_file() and path.resolve().is_relative_to(native),'Original native module missing or outside declared directory: '+name)
        cached=sys.modules.get(name)
        if cached is not None:
            require(getattr(cached,'__file__',None) is not None and Path(cached.__file__).resolve()==path.resolve(),
                'Cached native module origin mismatch: '+name)
    old_path=sys.path[:];old_package_path=package.__path__;old_modules=set(sys.modules)
    old_attributes={k:v for k,v in vars(package).items() if isinstance(v,ModuleType)}
    old_bytecode=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        sys.path[:0]=[str(native)]
        package.__path__=[str(native/'cvsrffi'),*old_package_path]
        importlib.invalidate_caches()
        for name,path in origins.items():
            search=[str(native/'cvsrffi')] if '.' in name else [str(native)]
            spec=PathFinder.find_spec(name,search)
            require(spec is not None and spec.origin is not None and Path(spec.origin).resolve()==path.resolve(),
                'Native module import origin mismatch: '+name)
            module=importlib.import_module(name)
            require(getattr(module,'__file__',None) is not None and Path(module.__file__).resolve()==path.resolve(),
                'Loaded native module origin mismatch: '+name)
            symbol=getattr(module,NATIVE_TYPES[name],None)
            require(isinstance(symbol,type) and symbol.__module__==name,'Original native checkpoint type missing: '+name)
        yield dict(native_code_ref=str(declared),resolved_native_code=str(native),
            module_files={name:str(Path(sys.modules[name].__file__).resolve()) for name in NATIVE_TYPES},
            scope='ORIGINAL_PRODUCER_TYPES_FOR_CHECKPOINT_DESERIALIZATION_ONLY')
    finally:
        sys.path[:]=old_path;package.__path__=old_package_path;sys.dont_write_bytecode=old_bytecode
        # Remove newly imported original-release modules, including local helper
        # dependencies, while preserving every module that existed on entry.
        for name in set(sys.modules)-old_modules:
            module=sys.modules.get(name);origin=getattr(module,'__file__',None)
            if origin is not None and Path(origin).resolve().is_relative_to(native):sys.modules.pop(name,None)
        for name,value in list(vars(package).items()):
            if isinstance(value,ModuleType) and name not in old_attributes and value.__name__ not in sys.modules:
                delattr(package,name)
        for name,value in old_attributes.items():setattr(package,name,value)
        importlib.invalidate_caches()


def completion_bytes(value,fixed_file_bytes):
    """Resolve JSON's own byte count before its sole exclusive write."""
    result=dict(value,completion_file_bytes=0,packet_total_file_bytes=fixed_file_bytes)
    for _ in range(16):
        encoded=encode_json(result);length=len(encoded)
        if result['completion_file_bytes']==length and result['packet_total_file_bytes']==fixed_file_bytes+length:return encoded,result
        result.update(completion_file_bytes=length,packet_total_file_bytes=fixed_file_bytes+length)
    raise ValueError('Completion byte-count representation did not stabilize')


def export_packet(*,checkpoint,binding,provenance,output,native_code=None):
    """Export a new immutable directory; preserve any owned partial output."""
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    require(sys.byteorder=='little','Packet byte order requires a little-endian host')
    metadata,source=bound_metadata(binding,provenance)
    checkpoint=Path(checkpoint);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();phase='checkpoint_identity'
    try:
        # Hash and deserialize the same open file, avoiding a pathname switch
        # between identity matching and head extraction. This is the existing
        # checkpoint identity binding, not a new packet receipt/hash chain.
        with checkpoint.open('rb') as stream:
            actual_digest=checkpoint_digest(stream)
            require(actual_digest==metadata.checkpoint_sha256,'Actual checkpoint SHA256 differs from the bound original')
            stream.seek(0);phase='native_checkpoint_types';tick=time.perf_counter()
            with native_checkpoint_types(native_code) as native_binding:
                phase='checkpoint_load'
                payload=torch.load(stream,map_location='cpu',weights_only=False)
            load_seconds=time.perf_counter()-tick
            checkpoint_file_bytes=stream.seek(0,2)
        phase='exact_head_extraction';weight,actual_key=extract_exact_weight(payload)
        # Validate through the real predictor ABI without running inference.
        GroundClassifierA(weight=weight,metadata=metadata)
        raw=bytes(weight.view(torch.uint8).reshape(-1).tolist())
        require(len(raw)==WEIGHT_BYTES,'Unexpected classifier byte count')
        phase='packet_write'
        with (out/WEIGHT_FILE).open('xb') as stream:stream.write(raw)
        packet=dict(schema=SCHEMA,head_metadata=asdict(metadata),
            weight=dict(file=WEIGHT_FILE,dtype='float32',shape=[GROUND_CLASS_COUNT,FEATURE_DIM],byte_order='little',
                numeric_bytes=WEIGHT_BYTES,actual_checkpoint_key=actual_key),
            class_row_mapping=[dict(row=i,class_id=name) for i,name in enumerate(metadata.ordered_classes)],
            binding_sources={name:binding[name] for name in ('class_row_order_source','factory_scale_source','corrections_source')},
            existing_source_only_provenance=source,
            provenance_scope='OWNER_SUPPLIED_EXISTING_SOURCE_ONLY_VERDICT_CONSUMED_NOT_REVERIFIED',
            inference=dict(dtype='float32',norm_eps=metadata.norm_eps,scale=metadata.scale,
                formula='F.linear(F.normalize(z_id.float(),dim=1,eps),F.normalize(weight.float(),dim=1,eps))*scale',
                tie_break='first_native_head_column',all_declared_ground_classes=True,logit_corrections='none'),
            export_measurements=dict(checkpoint_file_bytes=checkpoint_file_bytes,checkpoint_load_seconds=load_seconds,
                native_type_resolution=native_binding,
                torch_version=str(torch.__version__),encoder_constructed=False,encoder_executed=False,
                source_sample_rows_read=0,source_feature_rows_read=0,support_rows_read=0,query_rows_read=0,
                checkpoint_scores_inspected=False,actual_A_evaluated=False))
        with (out/METADATA_FILE).open('xb') as stream:stream.write(encode_json(packet))
        sizes={name:(out/name).stat().st_size for name in (WEIGHT_FILE,METADATA_FILE)}
        marker=dict(schema=SCHEMA,status=STATUS,checkpoint_sha256=metadata.checkpoint_sha256,
            files=sizes,weight_numeric_bytes=len(raw),export_seconds=time.perf_counter()-started,
            packet_byte_scope='ACTUAL_LOCAL_HEAD_WEIGHT_METADATA_AND_COMPLETION_FILES_NOT_ENCODER_OR_NETWORK_TRANSFER',
            actual_A_evaluated=False)
        encoded,marker=completion_bytes(marker,sum(sizes.values()))
        with (out/COMPLETE_FILE).open('xb') as stream:stream.write(encoded)
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file())==marker['packet_total_file_bytes'],
            'Measured packet file total mismatch')
        phase='packet_readback'
        loaded=load_packet(out)
        require(loaded.metadata==metadata and (out/WEIGHT_FILE).read_bytes()==raw,'Independent packet readback differs from exported head')
        return marker
    except Exception as exc:
        # Never replace, remove or silently retry the original checkpoint or any
        # file already written in this exclusive packet directory.
        failure=dict(schema=SCHEMA,status='GROUND_CLASSIFIER_A_PACKET_EXPORT_FAILED',phase=phase,
            error_type=type(exc).__name__,error=str(exc),checkpoint_sha256=metadata.checkpoint_sha256,
            completed_files=[p.name for p in out.iterdir() if p.is_file()])
        with (out/'export_failed.json').open('xb') as stream:stream.write(encode_json(failure))
        raise


def load_packet(packet):
    """Load only this small packet into the frozen native float32 predictor ABI."""
    root=Path(packet)
    require(not (root/'export_failed.json').exists(),'Failed packet is preserved and cannot be used')
    marker=json.loads((root/COMPLETE_FILE).read_text(encoding='utf-8'))
    require(marker['schema']==SCHEMA and marker['status']==STATUS and marker['actual_A_evaluated'] is False,'Incomplete packet')
    require(set(marker['files'])=={WEIGHT_FILE,METADATA_FILE},'Unexpected packet members')
    require({p.name for p in root.iterdir()}=={WEIGHT_FILE,METADATA_FILE,COMPLETE_FILE},'Packet directory member mismatch')
    for name,count in marker['files'].items():require((root/name).stat().st_size==count,'Packet file size mismatch')
    require((root/COMPLETE_FILE).stat().st_size==marker['completion_file_bytes'] and
        sum(p.stat().st_size for p in root.iterdir())==marker['packet_total_file_bytes'],'Packet measured total mismatch')
    value=json.loads((root/METADATA_FILE).read_text(encoding='utf-8'));head=value['head_metadata'];declared=value['weight']
    require(value['schema']==SCHEMA and declared['file']==WEIGHT_FILE and declared['dtype']=='float32'
        and declared['shape']==[6,160] and declared['numeric_bytes']==WEIGHT_BYTES and declared['byte_order']=='little'
        and sys.byteorder=='little','Unsupported weight representation')
    binding={name:head[name] for name in ('checkpoint_sha256','checkpoint_weight_key','ordered_classes','scale','norm_eps','feature_contract','logit_corrections')}
    binding.update(value['binding_sources']);metadata,_=bound_metadata(binding,value['existing_source_only_provenance'])
    require(head==json.loads(encode_json(asdict(metadata))),'Stored head metadata differs from native ABI declarations')
    require(metadata.checkpoint_sha256==marker['checkpoint_sha256'],'Packet checkpoint identity mismatch')
    require(value['class_row_mapping']==[dict(row=i,class_id=name) for i,name in enumerate(metadata.ordered_classes)],'Native class-row mapping mismatch')
    require(value['inference']['scale']==metadata.scale and value['inference']['norm_eps']==metadata.norm_eps
        and value['inference']['dtype']=='float32' and value['inference']['logit_corrections']=='none','Packet inference metadata mismatch')
    raw=(root/WEIGHT_FILE).read_bytes();require(len(raw)==WEIGHT_BYTES,'Head weight byte length mismatch')
    weight=torch.frombuffer(bytearray(raw),dtype=torch.float32).reshape(GROUND_CLASS_COUNT,FEATURE_DIM)
    return GroundClassifierA(weight=weight,metadata=metadata)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','binding','provenance','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-code',type=Path,help='Explicit original training release code directory for native pickle types')
    args=parser.parse_args()
    marker=export_packet(checkpoint=args.checkpoint,binding=json.loads(args.binding.read_text(encoding='utf-8')),
        provenance=json.loads(args.provenance.read_text(encoding='utf-8')),output=args.output,native_code=args.native_code)
    print(json.dumps(marker,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
