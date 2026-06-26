# Paper Baseline Rerun Report Template

## Run Identity

- Run ID:
- Timestamp:
- Operator:
- Repository commit:
- Dataset path:
- Conda/Python environment:

## Objective

- Target paper protocol:
- Hypothesis:
- Comparison target:

## Launch Record

| Field | Value |
|---|---|
| Command | |
| Working directory | |
| Log path | |
| Manifest path | |
| GPU allocation | |
| Seed | |
| WiSig protocol | |
| Receiver split | |
| Paper-window metric | |
| Satellite/stress view | |

## Verification Before Launch

| Check | Command | Result |
|---|---|---|
| Script syntax | | |
| Dry-run command | | |
| Dataset visibility | | |
| GPU/process occupancy | | |

## Same-Row Results

Do not report standalone maxima or minima as if they describe one experiment.
Keep each candidate's full row together.

| Candidate | Method | Protocol | Receiver split | Seed | Paper window | Main metric | Named-test metrics | Satellite/stress metrics | Log/checkpoint | Verdict |
|---|---|---|---|---:|---|---:|---|---|---|---|
| | | | | | | | | | | |

## Boundaries

- Dry-run is not training completion.
- Training completion is not deployment success.
- Satellite/stress evaluation is not real in-orbit validation.
- Claims must cite run directory, split, seed, and full same-row metrics.
