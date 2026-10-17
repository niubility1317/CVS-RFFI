"""Only existing labeled L and source V; U container is never read."""
from collections import Counter
from .design import RX


def select(packed, prepared, contract):
    if prepared.get('status') != 'SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared.get('target_access') is not False:
        raise ValueError('Matching existing source preparation required')
    for c in (packed['contract'], prepared['source_contract']):
        if c.get('role_ids') != contract['role_ids'] or c.get('native_role_comparison') != 'EXACT_MATCH':
            raise ValueError('Existing source role handles differ')
    roles = {}
    for key, handle, n, per_cell in (('labeled', 'L_s', 6300, 70), ('validation', 'V', 27000, 300)):
        role = {k: packed[key][k] for k in ('x', 'y', 'rx', 'day', 'ids')}
        if tuple(role['x'].shape) != (n, 2, 256):
            raise ValueError('Native complete source IQ256 required')
        ids = role['ids']
        if len(ids) != n or len(set(ids)) != n or set(ids) != set(contract['role_ids'][handle]):
            raise ValueError('Exact original role physical IDs required')
        for other in ('L_s', 'U_s', 'V'):
            if other != handle and set(ids) & set(contract['role_ids'][other]):
                raise ValueError('Physical roles must remain disjoint')
        y, rx, day = (role[k].tolist() for k in ('y', 'rx', 'day'))
        expected = Counter({(t, r, d): per_cell for t in range(6) for r in RX for d in (1, 2, 3)})
        if Counter(zip(y, rx, day)) != expected:
            raise ValueError('Every original source RX/TX/day cell required')
        roles[handle] = role
    return roles
