"""Trace generated conceptual artwork to SVG paths, retaining the original PNGs."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--vendor',type=Path)
    p.add_argument('--assets',nargs='*');args=p.parse_args()
    if args.vendor:sys.path.insert(0,str(args.vendor))
    import vtracer
    out=args.output
    for sub in ['png','svg','metadata']:(out/sub).mkdir(parents=True,exist_ok=True)
    names=args.assets or [p.stem for p in sorted(args.source.glob('*.png'))]
    for name in names:
        src=args.source/(name+'.png');png=out/'png'/(name+'.png');svg=out/'svg'/(name+'.svg')
        shutil.copy2(src,png);start=time.time()
        vtracer.convert_image_to_svg_py(str(png),str(svg),colormode='color',hierarchical='stacked',
            mode='spline',filter_speckle=0,color_precision=7,layer_difference=8,
            corner_threshold=60,length_threshold=4.0,max_iterations=10,splice_threshold=45,path_precision=2)
        text=svg.read_text(encoding='utf-8');paths=text.count('<path');images=text.count('<image')
        if paths<100 or images:raise RuntimeError(f'Invalid SVG {name}: {paths=} {images=}')
        metadata={'asset':name,'vector_paths':paths,'embedded_images':images,
            'bytes':svg.stat().st_size,'seconds':round(time.time()-start,1),
            'classification':'generative visual concept; semantic topology not verified'}
        (out/'metadata'/(name+'.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(metadata,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
