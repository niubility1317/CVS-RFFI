"""Explicit source mechanism evidence, with fresh evidence for each extension."""
import copy
import math


class SourceMechanismGate:
    """No default thresholds, automatic scientific qualification, or permanent latch.

    Scores are lower-is-better source risks/errors. The gate computes all three
    margins and difficult-group/LEO degradation from supplied paired measurements.
    A permitted module name must first be verified as terminal identity-only by
    the training integration; this metadata gate cannot inspect model topology.
    """
    def __init__(self, *, thresholds, stable_observations, min_samples,
                 source_freeze_id, source_frozen, terminal_identity_module):
        names = ('capability_min_improvement', 'necessity_min_gap',
                 'update_min_improvement', 'max_hard_group_degradation', 'max_leo_degradation')
        if (not isinstance(thresholds, dict) or set(thresholds) != set(names)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or v < 0 for v in thresholds.values())):
            raise ValueError('all nonnegative finite source thresholds must be explicit')
        if (source_frozen is not True or not isinstance(source_freeze_id, str)
                or not source_freeze_id.strip()):
            raise ValueError('explicit source parameter freeze required')
        for value in (stable_observations, min_samples):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('positive integer stability/sample requirements must be explicit')
        if (not isinstance(terminal_identity_module, str) or not terminal_identity_module.strip()
                or any(word in terminal_identity_module.lower() for word in ('shared', 'backbone', 'sinc'))
                or terminal_identity_module.lower() in ('all', '*', 'model', 'identity', 'cls_head')):
            raise ValueError('one specific terminal identity module required; shared/full backbone forbidden')
        self.config = dict(thresholds=copy.deepcopy(thresholds), stable_observations=stable_observations,
                           min_samples=min_samples, source_freeze_id=source_freeze_id,
                           source_frozen=True, terminal_identity_module=terminal_identity_module)
        self._frozen_config = copy.deepcopy(self.config)
        self.seen_observation_ids = set()
        self.scope = 'cls_head'
        self.streak = 0
        self.last_result = None

    @staticmethod
    def _finite(row, name):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('finite source metric required: ' + name)
        return float(value)

    def observe(self, evidence, *, scope='cls_head'):
        try:
            return self._observe(evidence, scope=scope)
        except (ValueError, TypeError, KeyError):
            # Malformed/stale/missing evidence cannot leave an earlier open
            # result visible to the training integration.
            self.streak = 0
            self.last_result = dict(authorized=False, passed=False, scope=scope,
                                    failed_conditions=['invalid_or_missing_source_evidence'])
            raise

    def _observe(self, evidence, *, scope):
        if self.config != self._frozen_config:
            raise ValueError('source-frozen gate configuration changed')
        if scope not in ('cls_head', self.config['terminal_identity_module']):
            raise ValueError('unregistered parameter expansion scope')
        if not isinstance(evidence, dict):
            raise ValueError('structured source evidence required')
        if evidence.get('source_role') != 'source_validation' or evidence.get('target_used') is not False:
            raise ValueError('only target-free source_validation evidence may enter this gate')
        observation_id = evidence.get('observation_id')
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ValueError('nonempty observation_id required')
        if observation_id in self.seen_observation_ids:
            raise ValueError('fresh observation required; old evidence cannot reopen or extend a gate')
        if evidence.get('scope') != scope:
            raise ValueError('evidence scope must match the proposed parameter scope')
        if evidence.get('source_freeze_id') != self.config['source_freeze_id']:
            raise ValueError('source evidence does not match the frozen thresholds')
        rows = {}
        for name in ('capability', 'necessity', 'update_value'):
            row = evidence.get(name)
            if not isinstance(row, dict):
                raise ValueError('missing source mechanism measurements: ' + name)
            count = row.get('sample_count')
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError('explicit source sample_count required')
            rows[name] = row
        capability = self._finite(rows['capability'], 'baseline_error') - self._finite(rows['capability'], 'full_error')
        necessity = self._finite(rows['necessity'], 'shuffled_tx_error') - self._finite(rows['necessity'], 'full_error')
        update = rows['update_value']
        improvement = self._finite(update, 'base_risk') - self._finite(update, 'joint_risk')
        train_ids, query_ids = update.get('training_physical_ids'), update.get('query_physical_ids')
        if not train_ids or not query_ids or set(train_ids) & set(query_ids):
            raise ValueError('update value needs nonempty disjoint source training/query physical records')
        if len(set(query_ids)) < update['sample_count']:
            raise ValueError('update sample_count exceeds unique independent query records')
        degradations = {}
        for name in ('hard_groups', 'leo_groups'):
            groups = update.get(name)
            if not isinstance(groups, dict) or not groups:
                raise ValueError('paired difficult-source and LEO groups must be measured')
            values = []
            for group in groups.values():
                if not isinstance(group, dict):
                    raise ValueError('group evidence must include paired risks and sample counts')
                count = group.get('sample_count')
                if isinstance(count, bool) or not isinstance(count, int) or count < self.config['min_samples']:
                    raise ValueError('insufficient difficult-source/LEO group evidence')
                values.append(self._finite(group, 'joint_risk') - self._finite(group, 'base_risk'))
            degradations[name] = max(values)
        thresholds = self.config['thresholds']
        metrics = dict(capability=capability, necessity=necessity, update_value=improvement)
        passed = {
            'capability': capability > thresholds['capability_min_improvement'],
            'necessity': necessity > thresholds['necessity_min_gap'],
            'update_value': improvement > thresholds['update_min_improvement']
                            and degradations['hard_groups'] <= thresholds['max_hard_group_degradation']
                            and degradations['leo_groups'] <= thresholds['max_leo_degradation']}
        for name, row in rows.items():
            passed[name] = bool(passed[name] and row['sample_count'] >= self.config['min_samples'])
        if self.scope != scope:
            self.streak = 0
        self.scope = scope
        self.streak = self.streak + 1 if all(passed.values()) else 0
        self.seen_observation_ids.add(observation_id)
        result = dict(source_role='source_validation', source_freeze_id=self.config['source_freeze_id'],
                      observation_id=observation_id, scope=scope,
                      passed=all(passed.values()), stable_observations=self.streak,
                      authorized=self.streak >= self.config['stable_observations'],
                      failed_conditions=[name for name in passed if not passed[name]])
        for name in passed:
            measured = {'improvement': metrics[name]}
            if name == 'update_value':
                measured.update(worst_hard_group_degradation=degradations['hard_groups'],
                                worst_leo_degradation=degradations['leo_groups'])
            result[name] = dict(passed=passed[name], sample_count=rows[name]['sample_count'], metrics=measured)
        self.last_result = copy.deepcopy(result)
        return result

    def request_extension(self, module_name, evidence):
        if module_name != self.config['terminal_identity_module']:
            self.streak = 0
            self.last_result = dict(authorized=False, passed=False,
                                    failed_conditions=['unregistered_parameter_scope'])
            raise ValueError('only the one registered terminal identity module may be requested')
        # observe enforces new scope, a fresh observation ID, and a new stability
        # streak. The old cls_head authorization is never inherited here.
        return self.observe(evidence, scope=module_name)

    def state_dict(self):
        return copy.deepcopy(dict(state_version=1, config=self.config, scope=self.scope,
                                  streak=self.streak, seen_observation_ids=self.seen_observation_ids,
                                  last_result=self.last_result))

    def load_state_dict(self, state):
        if state.get('state_version') != 1 or state.get('config') != self.config:
            raise ValueError('source mechanism gate resume configuration mismatch')
        for name in ('scope', 'streak', 'seen_observation_ids', 'last_result'):
            if name not in state:
                raise ValueError('incomplete source mechanism gate state')
        self.scope, self.streak = state['scope'], state['streak']
        self.seen_observation_ids = copy.deepcopy(state['seen_observation_ids'])
        self.last_result = copy.deepcopy(state['last_result'])
