"""Pure queue-state reconciliation for the three-GPU handoff."""


def partition(by, active, done, failed):
    errors = {(by[k]['origin'], by[k]['profile']) for k in failed}
    remaining = set(by) - set(active) - set(done) - set(failed)
    blocked = set()
    while True:
        newly = {k for k in remaining if
                 (by[k]['origin'], by[k]['profile']) in errors or
                 any(d in failed or d in blocked for d in by[k]['dependencies'])}
        if not newly:
            break
        remaining -= newly
        blocked |= newly
    return remaining, blocked, errors
