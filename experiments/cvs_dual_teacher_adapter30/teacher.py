"""Frozen same-IQ teachers; plain handles stay outside student state_dicts."""
from copy import deepcopy
import inspect
from pathlib import Path
import torch

CLASSES = ('14-10', '14-7', '20-15', '20-19', '6-15', '8-20')
CLEAN_RUN = '20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01'
SATELLITE_RUN = '20260930-phase1-native-baselines-practical-manysig-m5-r01'
SEED = 2026092701
COMMON_COSINE_SCALE = 30.


def calibration_interface():
    """Describe the explicit interface, without inventing a fitted reliability."""
    return dict(status='NOT_FITTED', source_only=True,
        allowed_inputs='Frozen legal source L labels or existing source diagnostic readouts only',
        fields=('clean_temperature', 'satellite_temperature', 'clean_teacher_scale',
                'clean_student_scale', 'satellite_teacher_scale', 'satellite_student_scale'),
        fixed_scale_conversion='Tc/student logits x1; CSIL native logits x6 gives cosine scale30',
        fitted_values=None, logits_sharpness_is_reliability=False,
        teacher_correctness_uses='teacher argmax and visible true L label only',
        target_access=False, automatic_fitting=False)


def _frozen(model):
    if (any(module.training for module in model.modules())
            or any(p.requires_grad or p.grad is not None for p in model.parameters())):
        raise ValueError('Teachers must remain eval with all parameters stop-gradient')


def _native_scale_metadata(clean, satellite, zero_bias_logits):
    """Inspect native classifier objects and the loaded CSIL function default."""
    head = clean.id_backbone.encoder.core.id_backbone.cls_head
    clean_scale = float(head.scale)
    if tuple(head.weight.shape) != (6, 160) or clean_scale != 30.:
        raise ValueError('Tc must use the original six-class joint160 cosine30 head')
    norm_mag = inspect.signature(zero_bias_logits).parameters['norm_mag'].default
    if (getattr(satellite, 'method', None) != 'csil'
            or not isinstance(satellite.fc_bf_fp, torch.nn.Linear)
            or satellite.fc_bf_fp.out_features != 6
            or tuple(satellite.fingerprints.shape) != (6, 6) or float(norm_mag) != 5.):
        raise ValueError('Actual CSIL linear feature projection/cosine fingerprint head differs')
    return dict(Tc=dict(native_classifier_type=type(head).__name__,
                        final_classifier='native cosine', native_cosine_scale=clean_scale,
                        native_class_independent_bias=0., common_cosine_scale=30.,
                        common_unit_logit_multiplier=1.),
                Ts=dict(native_classifier_type=type(satellite).__name__,
                        linear_feature_projection='fc_bf_fp; not the final identity logits',
                        final_classifier='zero_bias_logits projected features versus fingerprints',
                        native_formula='5 * cosine(projected_features, fingerprints) + 5',
                        native_cosine_scale=5., native_class_independent_bias=5.,
                        common_cosine_scale=30., common_unit_logit_multiplier=6.,
                        common_unit_bias=30., softmax_bias_cancels=True,
                        native_function_module=zero_bias_logits.__module__))


class FrozenTeachers:
    """Plain Python holder, never an nn.Module or a student checkpoint member.

    Native EMA deepcopy may share these fixed handles; it must not construct or
    update a new teacher. No IQ conversion, augmentation or calibration occurs.
    """
    def __init__(self, clean, satellite, identity):
        self.clean = clean.eval().requires_grad_(False)
        self.satellite = satellite.eval().requires_grad_(False)
        self._identity = deepcopy(identity)
        self._packets = dict(Tc=0, Ts=0)
        _frozen(self.clean)
        _frozen(self.satellite)

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def identity(self):
        return deepcopy(self._identity)

    def calibration_requirements(self):
        return calibration_interface()

    def stats(self):
        return dict(packets=dict(self._packets), state_registered_in_student=False,
                    optimizer_updates=0, EMA_updates=0, head_fit_updates=0)

    def assert_frozen(self):
        _frozen(self.clean)
        _frozen(self.satellite)

    def _logits(self, iq, role):
        teacher = self.clean if role == 'Tc' else self.satellite
        _frozen(teacher)
        if (not torch.is_tensor(iq) or iq.ndim != 3 or iq.shape[1:] != (2, 256)
                or not len(iq) or not iq.is_floating_point() or not bool(torch.isfinite(iq).all())
                or iq.device != next(teacher.parameters()).device):
            raise ValueError('Teacher must receive the exact same finite student IQ tensor/device')
        with torch.no_grad():
            output = (teacher.forward_identity_only(iq) if role == 'Tc' else teacher(iq))
            logits = output['tx_logits']
        if logits.shape != (len(iq), 6) or not bool(torch.isfinite(logits).all()):
            raise ValueError('Frozen native teacher must return six ordered finite logits')
        self._packets[role] += len(iq)
        return logits.detach()

    def clean_logits(self, clean_iq):
        """Tc on the same clean tensor used by the student's clean slot."""
        return self._logits(clean_iq, 'Tc')

    def satellite_logits(self, as_iq):
        """Ts on the same complete As tensor used by the student's As slot."""
        return self._logits(as_iq, 'Ts')

    def same_iq_logits(self, clean_iq, as_iq):
        return dict(clean=self.clean_logits(clean_iq), satellite=self.satellite_logits(as_iq))


def load(device='cpu'):
    """Use the already verified provenance loaders and preserve construction RNG."""
    from experiments.cvs_F_nontrue20 import design as original
    from experiments.cvs_F_nontrue20.model import own_model
    from experiments.cvs_F_nontrue20.teacher import load as load_satellite
    if tuple(original.CLASSES) != CLASSES or original.RUN != CLEAN_RUN:
        raise ValueError('Pinned source class order or Tc run differs')
    row = next(r for r in original.rows() if r['arm'] == 'B0' and r['model_seed'] == SEED)
    config = original.config(row)
    source = original.read(original.SOURCE)
    if (source['num_classes'] != 6 or source['source_rxs'] != [1, 3, 4, 6, 8]
            or source['source_days'] != [1, 2, 3]
            or ('classes' in source and tuple(source['classes']) != CLASSES)):
        raise ValueError('Pinned physical source/class contract differs')
    dev = torch.device(device)
    devices = list(range(torch.cuda.device_count())) if dev.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        # own_model verifies B0's actual FINAL20 checkpoint, frozen metadata,
        # scratch ancestry and exact source role_ids before strict state load.
        clean = own_model(config, device)
        # Original loader verifies CSIL source JSON/payload/role_ids/classes,
        # scratch last200 provenance, source qualification and strict state.
        satellite = load_satellite(device)
    from paper_reproduction.cvs_aligned.adv3b02_official_repo_ci import zero_bias_logits
    native = _native_scale_metadata(clean, satellite, zero_bias_logits)
    identity = dict(class_order=list(CLASSES), class_order_changed=False,
        class_order_authority='Pinned design.CLASSES and unchanged original strict source-role/default-head loaders',
        model_seed=SEED, common_cosine_scale=COMMON_COSINE_SCALE,
        Tc=dict(run_id=CLEAN_RUN, row_id='B0-s'+str(SEED), epoch=20, updates=4440,
                checkpoint=str(Path(config['output_root'])/'final_ssdg.pth'),
                selection='final_only', source_role='clean same-IQ', native=native['Tc']),
        Ts=dict(run_id=SATELLITE_RUN, row_id='csil-s'+str(SEED), epoch=200, updates=9800,
                checkpoint=str(original.TEACHER/'last.pt'), selection='fixed_last_epoch',
                source_role='same complete practical IQ', native=native['Ts']),
        source_contract=str(original.SOURCE), provenance='original verified loaders reused; no new inheritance',
        calibration=calibration_interface(), teacher_state_registered_in_student=False,
        target_access=False, new_head_training=False)
    return FrozenTeachers(clean, satellite, identity)
