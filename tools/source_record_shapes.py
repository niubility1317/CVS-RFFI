"""Inspect ndarray-like shape tuples without unpickling or reading large binary payloads."""
import argparse,collections,json,pickletools,struct
from pathlib import Path
LENGTHS={'BINBYTES':4,'BINBYTES8':8,'BYTEARRAY8':8,'SHORT_BINBYTES':1,'BINSTRING':4,'SHORT_BINSTRING':1,'BINUNICODE':4,'BINUNICODE8':8,'SHORT_BINUNICODE':1}
INTEGER={'INT','BININT','BININT1','BININT2','LONG','LONG1','LONG4'}
MEMO={'PUT','BINPUT','LONG_BINPUT','MEMOIZE','FRAME'}
def inspect(path):
 path=Path(path);size=path.stat().st_size;shapes=collections.Counter();payloads=collections.Counter();ints=[];ops=0;skipped=0;small=0;stopped=False
 with path.open('rb')as f:
  while True:
   raw=f.read(1)
   if not raw:break
   op=pickletools.code2op.get(raw.decode('latin1'))
   if op is None:raise ValueError('Unknown pickle opcode at '+str(f.tell()-1))
   name=op.name;ops+=1
   if name in ('STRING','UNICODE','FLOAT'):raise ValueError('Text payload protocol unsupported; do not read numerical content')
   if name=='BINFLOAT':f.seek(8,1);ints=[];continue
   if name in LENGTHS:
    width=LENGTHS[name];header=f.read(width)
    if len(header)!=width:raise ValueError('Truncated binary-field header')
    length=int.from_bytes(header,'little')
    if f.tell()+length>size:raise ValueError('Binary field exceeds file')
    if length>4096:f.seek(length,1);skipped+=length;payloads[length]+=1
    else:f.read(length);small+=length
    ints=[];continue
   arg=op.arg.reader(f)if op.arg else None
   if name in INTEGER and type(arg)is int:ints.append(arg)
   elif name in MEMO:pass
   elif name in ('TUPLE1','TUPLE2','TUPLE3','TUPLE'):
    count={'TUPLE1':1,'TUPLE2':2,'TUPLE3':3}.get(name,len(ints));shape=tuple(ints[-count:])if count else ()
    if len(shape)==count and 2<=len(shape)<=8 and all(0<x<=10**9 for x in shape):shapes[shape]+=1
    ints=[]
   else:ints=[]
   if name=='STOP':stopped=True;break
  consumed=f.tell()
 if not stopped or consumed!=size:raise ValueError('Expected one complete pickle; do not infer shapes from partial input')
 return dict(status='METADATA_INSPECTED',file_bytes=size,opcode_count=ops,shape_tuple_candidates=[dict(shape=list(k),count=v)for k,v in sorted(shapes.items())],skipped_large_binary_fields=[dict(bytes=k,count=v)for k,v in sorted(payloads.items())],skipped_binary_bytes=skipped,small_binary_metadata_bytes_read=small,unpickling_executed=False,numeric_arrays_constructed=False,large_numeric_payload_bytes_read=0,model_forwards=0,training_updates=0,note='Shape candidates are opcode metadata, not proof of time continuity or a data validation receipt.')
def main():
 p=argparse.ArgumentParser();p.add_argument('pickle',type=Path);a=p.parse_args();print(json.dumps(inspect(a.pickle),ensure_ascii=False))
if __name__=='__main__':main()
