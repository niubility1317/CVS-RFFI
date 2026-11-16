import unittest
from experiments.pamf_repl_attribution_factorial.analyze_source_proxy import analyze


def result_fixture():
    def relation(c):return dict(cosine=c,dot=c*2,right_norm=2.,left_norm=1.)
    settings={}
    for setting in ('ten_class','source_rx_6'):
        pairs=[];extremes=[]
        for i in range(8):
            options=[]
            for j in range(8):
                if i==j:continue
                entry=dict(labeled_offset=i,unlabeled_offset=j,proxy_CE_head=relation(float(j)),
                           actual_full_L_vs_raw_U={g:relation((j-3)/10) for g in ('t2','cls_head','other')})
                options.append(entry);pairs.append(entry)
            extremes.append(dict(labeled_offset=i,lowest=options[0],highest=options[-1]))
        settings[setting]=dict(cross_batch_pairs=pairs,per_L_extremes=extremes)
    return dict(status='SOURCE_CE_HEAD_PROXY_SCREEN_COMPLETE',U_truth_reads=0,V_reads=0,target_reads=0,
                original_parameter_updates=0,new_training=False,fixed_offsets=list(range(8)),proxy_screen=settings)


class SourceProxyAnalysisTests(unittest.TestCase):
    def test_paired_effect_and_complete_source(self):
        result=analyze(result_fixture())['settings']['ten_class']
        self.assertEqual(result['ordered_cross_pairs'],56)
        self.assertEqual(result['groups']['t2']['highest_more_aligned'],8)
        self.assertAlmostEqual(result['groups']['t2']['paired_cosine_difference_mean'],.675)

    def test_incomplete_matrix_and_truth_reads_rejected(self):
        result=result_fixture();result['target_reads']=1
        with self.assertRaisesRegex(ValueError,'source-only'):analyze(result)
        result=result_fixture();result['proxy_screen']['ten_class']['cross_batch_pairs'].pop()
        with self.assertRaisesRegex(ValueError,'comparison'):analyze(result)


if __name__=='__main__':unittest.main()
