import unittest


class SourceProxyDiagnosticTests(unittest.TestCase):
    def test_proxy_ranking_and_full_gradient_disagreement(self):
        import torch
        from experiments.pamf_repl_attribution_factorial.source_proxy_diagnostic import summarize
        named=[('model.cls_head.weight',torch.zeros(2)),('model.t2.weight',torch.zeros(2)),
               ('model.other.weight',torch.zeros(2))]
        def gradients(head,t2):
            return (torch.tensor(head),torch.tensor(t2),torch.tensor(t2))
        rows=[]
        for offset,u_head,u_full in ((0,[0.,1.],[0.,1.]),(1,[-1.,0.],[1.,0.]),(2,[1.,0.],[-1.,0.])):
            rows.append(dict(setting='ten_class',offset=offset,named=named,
                L_CE=gradients([1.,0.],[1.,0.]),L_full=gradients([1.,0.],[1.,0.]),
                U_CE=gradients(u_head,[0.,0.]),U_full=gradients(u_head,u_full)))
        result=summarize(rows)['ten_class']
        self.assertEqual(len(result['cross_batch_pairs']),6)
        rank=result['per_L_extremes'][0]
        self.assertEqual(rank['highest']['unlabeled_offset'],2)
        self.assertEqual(rank['lowest']['unlabeled_offset'],1)
        self.assertEqual(rank['highest']['proxy_CE_head']['cosine'],1.)
        self.assertEqual(rank['highest']['actual_full_L_vs_raw_U']['t2']['cosine'],-1.)
        self.assertEqual(rank['lowest']['actual_full_L_vs_raw_U']['t2']['cosine'],1.)
        self.assertTrue(all(r['labeled_offset']!=r['unlabeled_offset'] for r in result['cross_batch_pairs']))

    def test_insufficient_distinct_or_defined_candidates_rejected(self):
        import torch
        from experiments.pamf_repl_attribution_factorial.source_proxy_diagnostic import summarize
        named=[('model.cls_head.weight',torch.zeros(2))]
        row=dict(setting='ten_class',offset=0,named=named,L_CE=(torch.ones(2),),
                 L_full=(torch.ones(2),),U_CE=(torch.zeros(2),),U_full=(torch.ones(2),))
        with self.assertRaisesRegex(ValueError,'distinct'):
            summarize([row,dict(row)])
        with self.assertRaisesRegex(ValueError,'defined'):
            summarize([dict(row,offset=i) for i in range(3)])


if __name__=='__main__':unittest.main()
