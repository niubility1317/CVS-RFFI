"""Copy bounded source-only implementation dependencies into isolated Git carrier."""
from pathlib import Path
import shutil

ROOT=Path('E:/type10-7')
WT=ROOT/'github_publish/CVS-RFFI-repo/.worktrees/adv3b02-xuc-fusion-design-20260913'
DEST=WT/'experiments/adv3b02_xuc'
A1=ROOT/'code/snapshots/a1_fast_v2_20260909_wt'
GAME=ROOT/'code/snapshots/core90_game_20260911_wt'
U=ROOT/'code/snapshots/core90_cross_response_v2_20260911_wt'

def copy_tree(source,target):
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in {'.py','.json','.txt','.yaml','.yml'} and '__pycache__' not in path.parts:
            dest=target/path.relative_to(source)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)

if __name__=='__main__':
    if DEST.exists(): raise FileExistsError('Do not overwrite the implementation carrier')
    (DEST/'code').mkdir(parents=True)
    for path in (A1/'code').glob('*.py'):
        shutil.copy2(path,DEST/'code'/path.name)
    copy_tree(A1/'code/cvsrffi',DEST/'code/cvsrffi')
    copy_tree(A1/'code/SSDG',DEST/'code/SSDG')
    copy_tree(GAME/'code/cvsrffi/game_tracking',DEST/'code/cvsrffi/game_tracking')
    copy_tree(U/'code/cvsrffi/cross_response',DEST/'code/cvsrffi/cross_response')
    (DEST/'code/configs').mkdir()
    shutil.copy2(GAME/'code/configs/core90_game_historical_argv.json',DEST/'code/configs/core90_game_historical_argv.json')
    (DEST/'configs').mkdir()
    shutil.copy2(A1/'configs/a1_fast_matched_core90_20260908.json',DEST/'configs/a1_fast_matched_core90_20260908.json')
    (DEST/'code/scripts').mkdir()
    for name in ['run_a1_fast_matched_core90.py','run_a1_fast_selected_adv3b02.py','run_a1_scratch_no_checkpoint.py','run_a1_fast_v2.py','run_a1_ecrs_cross_rx.py']:
        shutil.copy2(A1/'code/scripts'/name,DEST/'code/scripts'/name)
    print('Source-only carrier created',DEST)
