"""Three registered arms on the historical WiSig Main split."""

from pathlib import Path

from experiments.pamf_wisig_branch_style import plan as matched

RUN = '20261125-phase1-pamf-wisig-preamble-bcd-m3-r01'
OWNER = 'codex-pamf-wisig-preamble-bcd-20261125'
ARMS = {'B': None, 'C': 'diagonal', 'D': 'low_rank'}


def row_id(arm):
    return matched.row_id(arm.lower())


def source_config(runtime, context, arm):
    c = matched.source_config(runtime, context, 'time')
    row = row_id(arm)
    c.update(run_id=RUN, row_id=row, preamble_bcd_arm=arm,
             output_root=str(Path(runtime['run_root']) / 'F' / row / 'source'),
             repl_row=dict(c['repl_row'], row_id=row))
    return validate_config(c)


def validate_config(c):
    arm = c.get('preamble_bcd_arm')
    if arm not in ARMS or c['run_id'] != RUN or c['row_id'] != row_id(arm):
        raise ValueError('Registered WiSig B/C/D arm required')
    matched.validate_config(dict(c, run_id=matched.RUN, row_id=matched.row_id('time')))
    if c['branch_style_arm'] != 'time' or c['repl_row']['row_id'] != c['row_id']:
        raise ValueError('Original temporal MixStyle and same-row REPL required')
    return c
