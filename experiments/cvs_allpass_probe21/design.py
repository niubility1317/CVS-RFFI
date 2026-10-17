"""Pure synthetic G0: fixed engineering falsification, no identity inference."""
from copy import deepcopy
RUN = "20261017-diagnostic-FcrossRX-allpass-synthetic-m1-r01"
SEED = 202610172101
LENGTH = 256
CONTEXT = 32
MAX_ALPHA = 0.04
ALPHAS = (-MAX_ALPHA, 0.0, MAX_ALPHA)
FAMILIES = ("white", "colored", "qpsk", "chirp", "periodic", "edge_burst", "amplitude_step", "edge_impulse")
LIMITS = dict(magnitude_error=1e-12, relative_change=0.1, relative_power_error=0.1,
              dense_error=5e-6, boundary_RMS=0.05, pooled_boundary_fraction=0.1,
              interior_error=1e-6, nonlinear_phase_fraction=0.1)
def config():
    return deepcopy(dict(run_id=RUN, stage="synthetic_G0_only", seed=SEED, length=LENGTH,
        reflection_context=CONTEXT, alphas=list(ALPHAS), families=list(FAMILIES),
        cases=32, output_views=96, limits=LIMITS, new_training_epochs=0,
        NN=0, IQ=0, checkpoint_reads=0, updates=0, target_access=False,
        model=None, labels=None, hardware_fit=False, deployment_change=False,
        on_fail="Close fixed construction/bound/context/fixtures/seed/criteria;no source or training.",
        on_pass="Engineering only;separately preregister legal frozen-default-head source identity checks."))
