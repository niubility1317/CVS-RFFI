"""Reject retired numerical variants of the author MoPC optimizer."""


def policy(job, group):
    version=job.get('mopc_stabilization','none')
    if version=='none':return None
    raise ValueError('MoPC stabilization variants have been retired')


def arm(original, version):
    if version=='none':return original
    raise ValueError('MoPC stabilization variants have been retired')
