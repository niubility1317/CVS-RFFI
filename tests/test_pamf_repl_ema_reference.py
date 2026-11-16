import unittest


class EMAReferenceTests(unittest.TestCase):
    def test_ema_reference_is_used_by_real_adamw_step(self):
        from test_pamf_optimizer_reference import make,grad
        from experiments.cric_dura_comparison import f_optimizer
        from experiments.pamf_ssl_research.optimizer_step_guard import installed
        from experiments.pamf_repl_ema_reference.objective import EMAReferenceGuard
        import torch
        model=make();opt=f_optimizer.build(model,model.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
        named=list(model.named_parameters());guard=EMAReferenceGuard(None,.25,.9);prior=None
        with installed():
            for index in range(2):
                full=grad(model,1.+index)
                guard.capture(model,named,full,full,tuple(None for _ in named))
                if prior is not None:
                    for current,old,new in zip(full,prior,guard.gradient_ema):
                        torch.testing.assert_close(new,.9*old+.1*current)
                prior=tuple(g.clone() for g in guard.gradient_ema)
                for parameter,value in zip(model.parameters(),grad(model,-2.-index)):
                    parameter.grad=value.clone()
                f_optimizer.clip_physical(opt,model.parameters(),5.)
                self.assertEqual(guard.reference_meta['reference_loss'],'segment_local_full_L_gradient_EMA')
                opt.step()
            self.assertEqual(guard.state_dict()['reference_gradient_EMA_updates'],2)
            self.assertEqual(guard.state_dict()['calls'],2)

    def test_detached_gradient_ema_and_missing_gradient_decay(self):
        import torch
        from experiments.pamf_repl_ema_reference.objective import update_reference_ema
        p=torch.tensor([2.],requires_grad=True);q=torch.tensor([4.],requires_grad=True)
        first=update_reference_ema(None,(p,None),.9)
        second=update_reference_ema(first,(q,torch.tensor([5.])),.9)
        torch.testing.assert_close(first[0],torch.tensor([2.]))
        torch.testing.assert_close(second[0],torch.tensor([2.2]))
        torch.testing.assert_close(second[1],torch.tensor([.5]))
        third=update_reference_ema(second,(None,None),.9)
        torch.testing.assert_close(third[0],torch.tensor([1.98]))
        torch.testing.assert_close(third[1],torch.tensor([.45]))
        self.assertFalse(second[0].requires_grad)
        with self.assertRaises(ValueError):update_reference_ema(first,(p,),1.)

    def test_matrix_scopes_and_ratios_match_full_reference(self):
        from experiments.pamf_repl_ema_reference import plan
        from experiments.pamf_repl_step_guard import plan as original
        old=original.RUN,original.RECIPES
        with plan.scoped():
            rows=original.candidate_rows();self.assertEqual(len(rows),8)
            self.assertEqual({(x['weight'],x['reference_cap'],x['reference_beta']) for x in plan.RECIPES.values()},
                             {(w,c,.9) for w in (3.,6.) for c in (.25,.5)})
            for row in rows:
                with self.assertRaises(FileNotFoundError):original.configuration({},row,'/missing','commit')
        self.assertEqual(original.RUN,old[0]);self.assertIs(original.RECIPES,old[1])


if __name__=='__main__':unittest.main()
