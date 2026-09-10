from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.runtime import train

if __name__=='__main__':
    raise SystemExit(train(parse_args()))
