"""Download pinned sources without editing existing upstream checkouts."""
import argparse
import subprocess
from pathlib import Path
from runtime import verify_source, PINS

p=argparse.ArgumentParser(); p.add_argument('--root',default='local_artifacts/external_refs')
args=p.parse_args(); root=Path(args.root); root.mkdir(parents=True,exist_ok=True)
for method,url in [('ISSL','https://github.com/Simple-up/ISSL.git'),
                   ('LoRa_RFFI','https://github.com/gxhen/LoRa_RFFI.git'),
                   ('LoRa_RFFI_Torch','https://github.com/PoisonT2/RFFI_Torch.git')]:
    dest=root/(method+'_20260930')
    if not dest.exists():
        subprocess.run(['git','clone','--no-checkout','--depth','1',url,str(dest)],check=True)
        subprocess.run(['git','-C',str(dest),'fetch','--depth','1','origin',PINS[method]],check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',PINS[method]],check=True)
    verify_source(dest,method)
    print('VERIFIED',method,PINS[method],dest)
