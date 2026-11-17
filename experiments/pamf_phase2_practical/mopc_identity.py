"""Keep retired MoPC results out of newly assembled comparisons."""


def require_author_mopc_row(row):
    if row.get('stage') != 'C' or 'mopc_hr' not in str(row.get('arm', '')).lower():
        return
    arm = str(row['arm']).lower()
    if 'clip' in arm:
        raise ValueError('Retired clipped MoPC row')
    resources = row.get('resources') or {}
    protocol = resources.get('C_protocol') or {}
    native = resources.get('native_resources') or {}
    if protocol:
        if protocol.get('mopc_version') != 'public_code_semantics_matched_IQ_extension':
            raise ValueError('Unverified or retired MoPC semantics')
        config = protocol.get('config') or {}
        if config.get('mopc_max_grad_norm') is not None or protocol.get('gradient_clip_max_norm') is not None:
            raise ValueError('Retired clipped MoPC row')
    elif native:
        if native.get('mopc_semantics') != 'public_code' or native.get('gradient_clip_max_norm') is not None:
            raise ValueError('Unverified or retired MoPC semantics')
    else:
        raise ValueError('MoPC version provenance is missing')
