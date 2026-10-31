"""Recognize other Phase3 restoration GPU entrypoints for shared admission."""


def is_restoration_gpu_worker(command):
    parts=command.split()
    for index,part in enumerate(parts):
        prefix='experiments.pamf_phase3_restoration.'
        if part.startswith(prefix):name=part[len(prefix):]
        elif '/pamf_phase3_restoration/' in part and part.endswith('.py'):name=part.rsplit('/',1)[-1][:-3]
        else:continue
        following=parts[index+1:]
        if name=='train':return True
        if name in ('historical_eval','phase3_eval'):return bool(following and following[0]=='predict')
        if name in ('cache','cost_probe'):return bool(following and following[0]=='worker')
    return False
