"""Materialize only five Python files from the pinned public author Git tree."""
import argparse,pathlib,subprocess
COMMIT='0090f4ba1d5cf76a4200a5be4830a9b08eb39699'
FILES=['main.py','util/CNNmodel.py','util/augmentation.py','util/con_losses.py','util/get_dataset.py']
def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args()
    head=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    if head!=COMMIT:raise ValueError('Unexpected author revision')
    a.output.mkdir()
    for name in FILES:
        content=subprocess.check_output(['git','-C',str(a.upstream),'show',COMMIT+':2TWC_Version/'+name])
        path=a.output/name;path.parent.mkdir(exist_ok=True);path.write_bytes(content)
    print('Five pinned author Python files copied unchanged; no weights or data copied.')
if __name__=='__main__':main()
