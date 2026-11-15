"""Explicitly version numerical changes to MoPC training."""


def policy(job, group):
    version=job.get('mopc_stabilization','none')
    if version=='none':return None
    if version!='clip1000_v1':raise ValueError('Unsupported MoPC stabilization')
    valid=(group=='native_cil' and job.get('profile')=='mopc_hr') or \
          (group=='hybrid' and job.get('methods')==['mopc_hr'])
    if not valid:raise ValueError('Stabilization requires a MoPC-only task')
    return 1000.0


def arm(original, version):
    if version=='none':return original
    if version!='clip1000_v1':raise ValueError('Unsupported MoPC stabilization')
    return original+'_clip1000_v1'
