"""Source freeze record only; no fitted classifier or new diagnostic forward."""
from . import design as d
from .provenance import source_proof


def run(row):
    c = d.config(next(r for r in d.rows() if r['row_id'] == row))
    p = source_proof(c)
    result = dict(status='SOURCE_MECHANISM_FROZEN', config=c,
                  mechanism=d.read(p / 'source_training_mechanism.json'),
                  default_head_only=True, head_fit_updates=0,
                  source_selection='fixed final20', target_access=False)
    out = p.parent / 'diagnostics'
    if (out / 'summary.json').exists():
        previous = d.read(out / 'summary.json')
        if previous != result:
            raise ValueError('Existing frozen source diagnostic differs; preserve it')
        return previous
    out.mkdir(exist_ok=False)
    d.write(out / 'summary.json', result)
    return result
