import random
import copy
import unittest

import numpy as np
import torch

from cvsrffi.cross_response.replay_audit import (
    capture_rng_state, capture_step_snapshot, disposable_counterfactual,
    first_divergence, grouped_source_risk, snapshot_training_state,
    weighted_gradient_audit,
)
from cvsrffi.cross_response.training import IndependentAuxiliaryTransaction


class ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(79)

    def test_owned_snapshot_and_first_divergence_stage(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        inputs = torch.randn(3, 2)
        logits = model(inputs)
        loss = logits.square().mean()
        loss.backward()
        kw = dict(inputs=inputs, batch_ticket={"physical_ids": [1, 2, 3]},
                  logits=logits, base_losses={"identity": loss}, model=model,
                  optimizer=optimizer, pseudo_state={"accepted": torch.tensor([1])})
        first = capture_step_snapshot(4, **kw)
        second = capture_step_snapshot(4, **kw)
        self.assertIsNone(first_divergence(first, second))
        second["unscaled_gradients"]["weight"][0, 0] += .25
        second["state"]["model"]["bias"] += 2
        found = first_divergence(first, second)
        self.assertEqual(found["stage"], "unscaled_gradients")
        self.assertEqual(found["path"], "unscaled_gradients.weight")
        self.assertEqual(found["step_id"], 4)
        self.assertGreater(found["max_abs"], .24)
        inputs.add_(10)
        self.assertFalse(torch.equal(first["inputs"], inputs))
        with self.assertRaises(ValueError):
            capture_step_snapshot(4, gradients_are_unscaled=False, **kw)

    def test_recursive_nonfinite_none_zero_and_tolerance(self):
        self.assertEqual(first_divergence({"p": None}, {"p": torch.zeros(1)})["reason"], "type")
        self.assertEqual(first_divergence(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["reason"], "nonfinite")
        self.assertIsNone(first_divergence(torch.tensor([1.]), torch.tensor([1.0001]), atol=.001))
        found = first_divergence({"a": [torch.zeros(1)]}, {"a": [torch.ones(1)]})
        self.assertEqual(found["path"], "a[0]")
        self.assertEqual(found["max_rel"], float("inf"))

    def test_weighted_reachability_excludes_classifier_keeps_zero(self):
        shared = torch.nn.Parameter(torch.tensor([1., 2.]))
        classifier = torch.nn.Parameter(torch.tensor([3.]))
        unused = torch.nn.Parameter(torch.tensor([4.]))
        losses = {"identity": shared.sum()+classifier.sum(),
                  "response": shared.sum()*2,
                  "decision": shared.sum()*0}
        report = weighted_gradient_audit(losses,
            [("repr", shared), ("classifier", classifier), ("unused", unused)],
            {"identity": 1., "response": .5, "decision": 1.},
            shared_parameter_names=["repr", "unused"])
        self.assertEqual(report["common_parameter_names"], ["repr"])
        self.assertEqual(report["reachability"]["decision"]["repr"], "zero")
        self.assertEqual(report["reachability"]["response"]["classifier"], "none")
        self.assertAlmostEqual(report["pairs"]["identity/response"]["cosine"], 1.)
        self.assertIsNone(report["pairs"]["identity/decision"]["cosine"])
        self.assertIsNone(shared.grad)

    def test_counterfactual_actual_adamw_preserves_caller_and_rng(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        x = torch.randn(4, 2)
        y = torch.tensor([0, 1, 0, 1])
        # Populate real AdamW moments, not an uninitialized-SGD approximation.
        model(x).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state = {"prototype": torch.zeros(2), "pseudo": [0], "x": x, "y": y}
        before = snapshot_training_state(model, optimizer, extra_state=state)
        seen = []
        def train(m, opt, s, response):
            seen.append((response, capture_rng_state()))
            opt.zero_grad(set_to_none=True)
            prediction = m(s["x"])
            loss = torch.nn.functional.cross_entropy(prediction, s["y"])
            if response:
                loss = loss + prediction[:, 0].mean()
            loss.backward()
            opt.step()
            s["prototype"].add_(1)
            s["pseudo"].append(9)
            random.random(); np.random.rand(); torch.rand(1)
            return {"steps": 1}
        def evaluate(m, s):
            return grouped_source_risk(m(s["x"]), s["y"], [1, 1, 3, 3],
                                       ["clean", "leo_a", "clean", "leo_a"])
        report = disposable_counterfactual(model, optimizer, train_step=train,
            evaluate=evaluate, train_physical_ids=[1, 2, 3, 4],
            eval_physical_ids=[5, 6, 7, 8], train_role="source_train",
            eval_role="source_validation", persistent_state=state)
        self.assertIsNone(first_divergence(before, snapshot_training_state(model, optimizer, extra_state=state)))
        self.assertIsNone(first_divergence(seen[0][1], seen[1][1]))
        self.assertGreater(report["base"]["delta_norm"], 0)
        self.assertEqual(report["initial"]["delta_norm"], 0)
        self.assertIsNotNone(first_divergence(report["base"]["parameter_delta"], report["base_response"]["parameter_delta"]))
        self.assertEqual(len(report["base"]["risk"]["groups"]), 4)
        self.assertEqual(report["base"]["post_state"]["optimizer"]["state"][0]["step"].item(), 2)

    def test_counterfactual_refuses_overlap_target_and_restores_on_failure(self):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters())
        def fail(*args):
            torch.rand(3)
            raise RuntimeError("injected")
        kw = dict(train_step=fail, evaluate=fail, train_physical_ids=[1],
                  eval_physical_ids=[2], train_role="source_train", eval_role="source_validation")
        before = capture_rng_state()
        with self.assertRaises(RuntimeError):
            disposable_counterfactual(model, opt, **kw)
        self.assertIsNone(first_divergence(before, capture_rng_state()))
        with self.assertRaises(ValueError):
            disposable_counterfactual(model, opt, **dict(kw, eval_physical_ids=[1]))
        with self.assertRaises(ValueError):
            disposable_counterfactual(model, opt, **dict(kw, eval_role="target"))

    def test_probe_uses_post_update_state_before_eval_mutations(self):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters())
        def train(m, opt, state, response):
            state['value'] = 7 if response else 5
        def evaluate(m, state):
            state['value'] = 100
            return {'ok': True}
        def probe(m, state):
            return {'value': state['value'], 'training': m.training}
        result = disposable_counterfactual(model, opt, train_step=train, evaluate=evaluate,
            train_physical_ids=[1], eval_physical_ids=[2], train_role='source_train',
            eval_role='source_validation', persistent_state={'value': 0}, probe=probe)
        self.assertEqual(result['base']['indirect_coupling_probe']['value'], 5)
        self.assertEqual(result['base_response']['indirect_coupling_probe']['value'], 7)
        self.assertTrue(result['base']['indirect_coupling_probe']['training'])

    def test_auxiliary_nan_inf_cannot_clip_skip_or_mutate_main_update(self):
        for nonfinite in (float('nan'), float('inf')):
            model = torch.nn.Linear(2, 2)
            reference = copy.deepcopy(model)
            main_opt = torch.optim.AdamW(model.parameters(), lr=.01)
            reference_opt = torch.optim.AdamW(reference.parameters(), lr=.01)
            head = torch.nn.Linear(2, 1)
            head_before = copy.deepcopy(head.state_dict())
            transaction = IndependentAuxiliaryTransaction(head.parameters(), lr=.01,
                weight_decay=.1, amp=False, max_grad_norm=1e-8)
            x = torch.randn(4, 2)
            out = model(x)
            out.square().mean().backward()
            # Detach is a second protection; the transaction itself asks only
            # for auxiliary gradients and never clips the main parameter set.
            report = transaction.step(head(out.detach()).sum()*nonfinite)
            main_opt.step()
            reference(x).square().mean().backward()
            reference_opt.step()
            self.assertEqual(report['auxiliary_step_applied'], 0.)
            self.assertIsNone(first_divergence(head_before, head.state_dict()))
            self.assertIsNone(first_divergence(model.state_dict(), reference.state_dict()))
            self.assertIsNone(first_divergence(main_opt.state_dict(), reference_opt.state_dict()))


if __name__ == "__main__":
    unittest.main()
