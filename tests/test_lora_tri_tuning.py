from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import tri_tuning
from experiments.pamf_lora_main.clean_aperture import ALL_CHIRPS, padded_crop
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.full_tri_repl_lora import clean_source_ce
from experiments.pamf_lora_main.packet_physics import chirp_samples
from experiments.pamf_lora_main.tri_branch_shapes import retain_projection
from experiments.pamf_lora_main import tri_clean95
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.repair_features import packet_phase_curvature
from experiments.pamf_ssl_research.hard_pl_objective import mechanism_loss
from experiments.pamf_ssl_research.source_receiver_extrapolation import extrapolate


class TriTuningTests(TestCase):
    def test_rejects_target_for_unselected_row_before_opening_inputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write(root/'selection.json', {'names': ['control']})
            with self.assertRaisesRegex(ValueError, 'source-frozen eligible'):
                tri_tuning.predict({'run_root':temp}, 'fusion192')

    def test_physical_ce_can_change_without_enabling_as_kl(self):
        a = torch.zeros(2, 10)
        args = (a, a, a, a+torch.arange(10), a, a, torch.tensor([0,1]),
                torch.ones(2,dtype=torch.bool), torch.tensor([True, False]), torch.tensor(0.))
        base, info = mechanism_loss(*args)
        modified, other = mechanism_loss(*args, physical_weight=2.)
        self.assertGreater(modified.item(), base.item())
        self.assertGreater(info['As_KL'].item(), 0.)
        self.assertEqual(other['As_KL_weight'], 0.)

    def test_extrapolation_preserves_registered_width_and_rx_constraint(self):
        z = torch.stack([torch.zeros(192), torch.ones(192)])
        ids = torch.tensor([0,1])
        same = torch.zeros(2,dtype=torch.long)
        mask = torch.ones(2,dtype=torch.bool)
        partner = ids.flip(0)
        result = extrapolate(z, same, mask, ids, partner, mask, .25, width=192)
        self.assertTrue(torch.equal(result[0], torch.full((192,), -.25)))
        with self.assertRaisesRegex(ValueError, 'different-sourceRX'):
            extrapolate(z, same, mask, same, partner, mask, .25, width=192)
        with self.assertRaises(ValueError):
            extrapolate(z, same, mask, ids, partner, mask, .25)

    def test_long_grid_retains_native_band_unavailable_to_256_points(self):
        timeline = np.arange(8192)/1_000_000
        tone = np.exp(2j*np.pi*170_000*timeline)
        low = chirp_samples(tone,samples=256)
        high = chirp_samples(tone,samples=512)
        self.assertEqual(low.shape,(6,256))
        self.assertEqual(high.shape,(6,512))
        self.assertLess(np.mean(np.abs(low)),.05)
        self.assertGreater(np.mean(np.abs(high)),.95)

    def test_wider_branch_pool_preserves_the_projected_mean_at_initialization(self):
        layer = torch.nn.Linear(3,5)
        values = torch.arange(48,dtype=torch.float32).reshape(2,3,8)
        old = layer(values.mean(dim=2))
        new = retain_projection(layer)(torch.nn.functional.adaptive_avg_pool1d(values,4).flatten(1))
        torch.testing.assert_close(old,new,atol=1e-5,rtol=1e-5)

    def test_vectorized_clean_grid_matches_the_registered_packet_readout(self):
        rng = np.random.default_rng(26)
        x = (rng.normal(size=(3,8192))+1j*rng.normal(size=(3,8192))).astype(np.complex64)
        for n in (192,256,384,512):
            actual = tri_clean95.clean_iq(x,n)
            expected = np.stack([chirp_windows(raw,samples=n) for raw in x])
            np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-7)

    def test_clean_predictor_blocks_unselected_rows_without_target_handle(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write(root/'selection.json', {'names':['control']})
            with self.assertRaisesRegex(ValueError, 'source-frozen'):
                tri_clean95.predict({'run_root':temp},'iq512')

    def test_robust_study_only_predicts_for_the_source_champion_at_98_percent(self):
        for n, correct, expected in ((12000,11759,[]), (11999,11760,[]),
                                     (12000,11760,['joint_rx'])):
            with self.subTest(n=n, correct=correct), TemporaryDirectory() as temp:
                root=Path(temp)/'run'
                rt={'run_root':str(root),'scratch':True,'robust_study':True,
                    'fixed_clean_cache':'source-only'}
                predicted=[]

                def schedule(_,jobs,launch):
                    for kind,name in jobs:
                        if kind=='predict':
                            predicted.append(name)
                            continue
                        record={'n':n,'correct':correct if name=='joint_rx' else 11478,'nll':.1}
                        write(root/'F'/name/'complete.json',{'frozen':record})

                with patch.object(tri_clean95,'start',return_value=Mock(wait=lambda:0)), \
                     patch.object(tri_clean95,'schedule',side_effect=schedule):
                    tri_clean95.dispatch(rt)
                self.assertEqual(predicted,expected)
                self.assertEqual(read(root/'selection.json')['names'],expected)

    def test_physics_flag_reaches_source_workers_without_target_handle(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)/'run'
            rt={'run_root':str(root),'scratch':True,'physics_study':True,
                'fixed_clean_cache':'source-only','native_target_root':'private'}
            with patch.object(tri_clean95,'start',return_value=Mock(wait=lambda:0)) as start, \
                 patch.object(tri_clean95,'schedule') as schedule:
                def freeze(_,jobs,launch):
                    self.assertEqual(jobs,[('train','phase_physics')])
                    launch(rt,'train','phase_physics',0)
                    write(root/'F/phase_physics/complete.json',{'frozen':{'n':12000,'correct':11759,'nll':.1}})
                schedule.side_effect=freeze
                tri_clean95.dispatch(rt)
            for call in start.call_args_list:
                self.assertIn('physics_study',call.kwargs['extra_keys'])
                self.assertNotIn('native_target_root',call.kwargs['extra_keys'])
            self.assertEqual(read(root/'selection.json')['names'],[])

    def test_packet_phase_feature_removes_common_phase_and_linear_drift(self):
        t=torch.arange(8,dtype=torch.float64)
        base=torch.polar(1+.2*t,torch.sin(t)).unsqueeze(0)
        rotated=base*torch.polar(torch.ones_like(t),.4+.2*t)
        actual=packet_phase_curvature(rotated)
        self.assertEqual(actual.shape,(1,12))
        torch.testing.assert_close(actual,packet_phase_curvature(base))

    def test_packet_physics_cancels_common_gain_phase_and_frequency(self):
        from experiments.pamf_lora_main.repair_features import relative_packet_physics
        rng=torch.Generator().manual_seed(43)
        residual=torch.randn(2,8,384,dtype=torch.complex64,generator=rng)+1
        tone=torch.randn(2,8,generator=rng)*20
        dc=residual.mean(-1)
        expected=relative_packet_physics(tone,dc,residual)
        scaled=residual*(1.6+.7j)
        actual=relative_packet_physics(tone+326,scaled.mean(-1),scaled)
        self.assertEqual(actual.shape,(2,24))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        rows,epochs,u_batch=tri_clean95.study({'scratch':True,'physics_study':True})
        self.assertEqual((epochs,u_batch),(112,16))
        self.assertTrue(rows['phase_physics']['packet_physics'])

    def test_timing_views_use_neighbor_samples_within_every_physical_chirp(self):
        center=padded_crop(ALL_CHIRPS)[:,125:1125]
        for offset in (-2,2):
            shifted=padded_crop(ALL_CHIRPS,offset)[:,125:1125]
            np.testing.assert_array_equal(shifted,center+offset)
            self.assertTrue((shifted[:,0]>=np.arange(8)*1024).all())
            self.assertTrue((shifted[:,-1]<(np.arange(8)+1)*1024).all())

    def test_chirp_readout_starts_uniform_and_learns_position_specific_scores(self):
        gate=tri_clean95.ChirpReadout(160,10,8)
        x=torch.randn(4,8,160,generator=torch.Generator().manual_seed(32))
        weight=torch.randn(10,160,generator=torch.Generator().manual_seed(6))
        correction=gate(x,weight)
        self.assertEqual(tuple(correction.shape),(4,10))
        torch.testing.assert_close(correction,torch.zeros_like(correction))
        correction.sum().backward()
        self.assertGreater(float(gate.position.grad.abs().sum()),0.)
        self.assertGreater(float(gate.gate.weight.grad.abs().sum()),0.)
        rows,epochs,u_batch=tri_clean95.study({'scratch':True,'chirpread_study':True})
        self.assertEqual((epochs,u_batch),(112,16))
        self.assertTrue(rows['phase_chirpread']['chirp_readout'])

    def test_coverage_uses_unique_source_indices_for_all_repl_epochs(self):
        rows,epochs,u_batch=tri_clean95.study({'scratch':True,'coverage_study':True})
        row=rows['phase_coverage']
        self.assertTrue(row['cover_u'])
        rng=torch.Generator().manual_seed(tri_clean95.SEED)
        options=dict(device='cpu',generator=rng)
        actual=tri_clean95.distinct_u_batches(25200,epochs-row['warmup'],88,u_batch,**options)
        self.assertEqual(tuple(actual.shape),(16,88,16))
        self.assertEqual(actual.unique().numel(),22528)
        self.assertTrue(0<=actual.min()<=actual.max()<25200)
        rng.manual_seed(tri_clean95.SEED)
        torch.testing.assert_close(actual,tri_clean95.distinct_u_batches(25200,16,88,16,**options))
        with self.assertRaisesRegex(ValueError,'Unique source U budget'):
            tri_clean95.distinct_u_batches(100,16,88,16,**options)

    def test_repl_rtl_gate_only_tightens_the_registered_source_receiver(self):
        from experiments.pamf_lora_main.full_tri_repl_lora import repl_acceptance
        rtl=SOURCE.index('rtl_1')
        confidence=torch.tensor([.949,.95,.974,.975,.95,.974,.975])
        receiver=torch.tensor([rtl,rtl,rtl,rtl,0,1,2])
        expected=torch.tensor([False,False,False,True,True,True,True])
        torch.testing.assert_close(repl_acceptance(confidence,receiver),confidence>=.95)
        torch.testing.assert_close(repl_acceptance(confidence,receiver,.975),expected)
        rows,epochs,batch=tri_clean95.study({'scratch':True,'rtlgate_study':True})
        self.assertEqual((epochs,batch),(112,16))
        self.assertEqual(rows['phase_rtlgate']['rtl_repl_threshold'],.975)

    def test_tail_average_matches_equal_snapshot_weights_and_preserves_integer_buffers(self):
        average=None
        for count,value in enumerate((2.,-2.,3.,7.),start=1):
            state={'model.weight':torch.tensor([value]),'tracked':torch.tensor(count)}
            average=tri_clean95.accumulate_state(average,state,count)
            state['model.weight'].zero_()
        torch.testing.assert_close(average['model.weight'],torch.tensor([2.5]))
        self.assertEqual(average['tracked'].item(),4)
        rows,epochs,batch=tri_clean95.study({'scratch':True,'tailaverage_study':True})
        self.assertEqual((epochs,batch),(112,16))
        self.assertEqual(rows['phase_tailaverage']['tail_average'],(100,104,108,112))

    def test_native_edge_aperture_preserves_center_and_source_sample_positions(self):
        from experiments.pamf_lora_main.clean_aperture import NATIVE_EDGE_POSITIONS, clean_iq
        random=np.random.default_rng(22)
        raw=(random.standard_normal((2,8192))+1j*random.standard_normal((2,8192))).astype(np.complex64)
        central=clean_iq(raw,384,ALL_CHIRPS)
        packed=clean_iq(raw,384,ALL_CHIRPS,native_edges=True)
        self.assertEqual(packed.shape,(2,8,2,408))
        np.testing.assert_array_equal(packed[...,:384],central)
        indices=np.arange(8)[:,None]*1024+np.asarray(NATIVE_EDGE_POSITIONS)
        received=packed[:,:,0,384:]+1j*packed[:,:,1,384:]
        scale=received/raw[:,indices]
        np.testing.assert_allclose(scale,np.broadcast_to(scale[:,0:1,0:1],scale.shape),atol=1e-6)
        with self.assertRaisesRegex(ValueError,'Native boundaries require'):
            clean_iq(raw,384,ALL_CHIRPS,offset=1,native_edges=True)
        key,layout=tri_clean95.cache_layout(384,ALL_CHIRPS,'source',native_edges=True)
        self.assertEqual(key,'384-all8-nativeedges')
        self.assertEqual(layout['native_edge_positions'],list(NATIVE_EDGE_POSITIONS))

    def test_native_phase_features_cancel_global_gain_and_phase(self):
        from experiments.pamf_lora_main.repair_features import NativeBoundaryPhase
        from experiments.pamf_lora_main.physical import ChirpResponse
        model=NativeBoundaryPhase(160)
        ref=ChirpResponse(samples=384).chirp
        a=torch.randn(3,8,2,408,generator=torch.Generator().manual_seed(47))
        baseline=model.features(a[...,:384],a[...,384:],ref)
        z=torch.complex(a[:,:,0],a[:,:,1])*(1.2+.55j)
        b=torch.stack((z.real,z.imag),dim=2)
        self.assertEqual(tuple(baseline.shape),(3,197))
        torch.testing.assert_close(model.features(b[...,:384],b[...,384:],ref),baseline,
                                   atol=2.e-6,rtol=2.e-6)

    def test_wider_native_boundaries_preserve_core_and_use_distinct_shapes(self):
        from experiments.pamf_lora_main.clean_aperture import clean_iq, native_edge_positions
        from experiments.pamf_lora_main.repair_features import NativeBoundaryPhase
        from experiments.pamf_lora_main.physical import ChirpResponse

        random=np.random.default_rng(31)
        raw=(random.standard_normal((2,8192))+1j*random.standard_normal((2,8192))).astype(np.complex64)
        central=clean_iq(raw,384,ALL_CHIRPS)
        ref=ChirpResponse(samples=384).chirp
        rows,epochs,batch=tri_clean95.study({'scratch':True,'nativewide_study':True})
        self.assertEqual((epochs,batch),(112,16))
        self.assertEqual(set(rows),{'phase_native16','phase_native24'})
        for width,feature_dim in ((16,261),(24,389)):
            with self.subTest(width=width):
                row=rows[f'phase_native{width}']
                self.assertTrue(row['native_edges'])
                self.assertEqual(row['edge_points'],width)
                packed=clean_iq(raw,384,ALL_CHIRPS,native_edges=True,edge_points=width)
                self.assertEqual(packed.shape,(2,8,2,384+2*width))
                np.testing.assert_array_equal(packed[...,:384],central)
                indices=np.arange(8)[:,None]*1024+np.asarray(native_edge_positions(width))
                received=packed[:,:,0,384:]+1j*packed[:,:,1,384:]
                scale=received/raw[:,indices]
                np.testing.assert_allclose(scale,np.broadcast_to(scale[:,0:1,0:1],scale.shape),atol=1e-6)
                x=torch.as_tensor(packed)
                model=NativeBoundaryPhase(160,edge_points=width)
                self.assertEqual(tuple(model.features(x[...,:384],x[...,384:],ref).shape),(2,feature_dim))
                key,layout=tri_clean95.cache_layout(384,ALL_CHIRPS,'source',native_edges=True,edge_points=width)
                self.assertEqual(key,f'384-all8-nativeedges{width}')
                self.assertEqual(layout['native_edge_positions'],list(native_edge_positions(width)))

    def test_native_lag_keeps_the_declared_boundary_shape_and_cancels_core_carrier(self):
        from experiments.pamf_lora_main.clean_aperture import native_edge_positions
        from experiments.pamf_lora_main.physical import ChirpResponse
        from experiments.pamf_lora_main.repair_features import NativeBoundaryPhase

        core_ref=ChirpResponse(samples=384).chirp
        carrier=.03
        shape=.10
        core_phase=torch.arange(384)*carrier/.384
        core_signal=(core_ref*torch.exp(1j*core_phase)).expand(2,8,-1)
        core=torch.stack((core_signal.real,core_signal.imag),dim=2)
        rows,epochs,u_batch=tri_clean95.study({'scratch':True,'native_lag_study':True})
        self.assertEqual((epochs,u_batch),(112,16))
        for points,lag,dimensions in ((24,8,277),(32,14,309)):
            with self.subTest(points=points,lag=lag):
                row=rows[f'phase_native{points}_lag{lag}']
                self.assertEqual((row['edge_points'],row['edge_lag']),(points,lag))
                edge=NativeBoundaryPhase(160,edge_points=points,lag=lag)
                native_t=torch.tensor(native_edge_positions(points))
                edge_signal=(edge.edge_reference.flatten()*torch.exp(1j*native_t*(carrier+shape))).expand(2,8,-1)
                iq=torch.stack((edge_signal.real,edge_signal.imag),dim=2)
                actual=edge.features(core,iq,core_ref)
                self.assertEqual(tuple(actual.shape),(2,dimensions))
                expected=torch.full_like(actual[:,:-21],np.sin(lag*shape))
                torch.testing.assert_close(actual[:,:-21],expected,atol=1e-5,rtol=1e-5)

    def test_native_receiver_loss_uses_the_identity_branch_with_cross_receiver_pairs(self):
        from types import SimpleNamespace
        from experiments.pamf_lora_main.physical import ChirpResponse
        from experiments.pamf_lora_main.repair_features import NativeBoundaryPhase

        boundary=NativeBoundaryPhase(16,edge_points=24)
        generator=torch.Generator().manual_seed(962)
        with torch.no_grad():
            boundary.project[-1].weight.copy_(torch.randn(16,96,generator=generator)*.02)
        trace=SimpleNamespace(time=torch.empty(384),reference=ChirpResponse(samples=384))
        model=SimpleNamespace(native_boundary=boundary,trace=trace)
        iq=torch.randn(4,8,2,432,generator=generator)
        ids=torch.tensor([0,0,1,1])
        rx=torch.tensor([0,1,0,1])
        loss,valid=tri_clean95.native_boundary_receiver_loss(model,iq,ids,rx)
        self.assertEqual(int(valid),4)
        self.assertGreater(float(loss.detach()),0.)
        loss.backward()
        self.assertTrue(torch.isfinite(boundary.project[-1].weight.grad).all())
        self.assertGreater(float(boundary.project[-1].weight.grad.abs().sum()),0.)
        missing,no_pairs=tri_clean95.native_boundary_receiver_loss(model,iq,ids,torch.zeros_like(rx))
        self.assertEqual((float(missing.detach()),int(no_pairs)),(0.,0))
        rows,epochs,u_batch=tri_clean95.study({'scratch':True,'native_rx_study':True})
        row=rows['phase_native24_rx']
        self.assertEqual((row['edge_points'],row['native_rx_weight'],epochs,u_batch),(24,.04,112,16))

    def test_extended_repl_uses_the_same_decay_then_stays_at_the_floor(self):
        import math

        train_steps=tri_clean95.STEPS
        original_end=112*train_steps
        warm=96*train_steps
        rows,epochs,u_batch=tri_clean95.study({'scratch':True,'native_repl32_study':True})
        row=rows['phase_native24_repl32']
        self.assertEqual((row['warmup'],epochs,u_batch),(96,128,16))
        self.assertEqual((row['decay_end_epoch'],row['select_from_epoch']),(112,116))
        self.assertEqual(row['edge_points'],24)
        factor=tri_clean95.scratch_repl_lr_factor
        for step in (0,warm-1,warm,warm+1,104*train_steps,original_end-1,original_end):
            previous=.25+.75*.5*(1+math.cos(math.pi*step/warm)) if step<warm else (
                1/30.+(.25-1/30.)*.5*(1+math.cos(math.pi*(step-warm)/(original_end-warm))))
            self.assertAlmostEqual(factor(step,warm,original_end),previous,places=14)
        self.assertAlmostEqual(factor(original_end+1,warm,original_end),1/30.,places=14)
        self.assertAlmostEqual(factor(epochs*train_steps,warm,original_end),1/30.,places=14)

    def test_native_fractional_shift_uses_one_time_coordinate_for_center_and_edges(self):
        from experiments.pamf_lora_main.clean_aperture import clean_iq, shift_native_packets

        native_time=np.arange(8192,dtype=np.float64)/1024
        curve=lambda t: (1+.03*t**2)+1j*(.04*t-.003*t**3)
        raw=curve(native_time).astype(np.complex64)[None,:]
        rows,epochs,u_batch=tri_clean95.study({'scratch':True,'native_jitter_study':True})
        row=rows['phase_native24_jitter']
        self.assertEqual((row['native_timing_offset'],row['edge_points'],epochs,u_batch),(.5,24,112,16))
        ordinary=clean_iq(raw,384,ALL_CHIRPS,native_edges=True,edge_points=24)
        views=tri_clean95.labeled_shift_views(raw,row)
        for shift,view in zip((-.5,.5),views,strict=True):
            shifted=shift_native_packets(raw,shift)
            reference=curve(native_time+shift/1024).astype(np.complex64)
            np.testing.assert_allclose(shifted[0,2:-2],reference[2:-2],rtol=1e-6,atol=1e-6)
            self.assertEqual(view.shape,(1,8,2,432))
            self.assertFalse(np.array_equal(view,ordinary))
            np.testing.assert_array_equal(view,clean_iq(shifted,384,ALL_CHIRPS,native_edges=True,edge_points=24))

    def test_group_residual_is_bounded_per_packet_without_touching_primary_logits(self):
        from experiments.pamf_lora_main.repair_features import residual_logit_scale
        core=torch.tensor([[0.,1.,2.],[0.,2.,4.],[5.,5.,5.]])
        delta=torch.tensor([[0.,4.,8.],[1.1,1.,.9],[0.,1.,2.]])
        scale=residual_logit_scale(core,delta,.35)
        torch.testing.assert_close(scale[:,0],torch.tensor([.0875,1.,0.]))
        shifted=residual_logit_scale(core+1.,delta-2.,.35)
        torch.testing.assert_close(shifted,scale)
        rows,epochs,batch=tri_clean95.study({'scratch':True,'groupcap_study':True})
        self.assertEqual((epochs,batch),(112,16))
        self.assertEqual(rows['phase_groupcap']['group_logit_limit'],.35)

    def test_grouped_features_are_initially_neutral_and_learnable(self):
        from experiments.pamf_lora_main.repair_features import PacketGroupFusion
        rng=torch.Generator().manual_seed(73)
        inputs=(torch.randn(5,8,160,generator=rng),torch.randn(5,36,generator=rng),
                torch.randn(5,160,generator=rng))
        for radial in (False,True):
            with self.subTest(radial=radial):
                branch=PacketGroupFusion(160,8,radial=radial)
                output=branch(*inputs)
                torch.testing.assert_close(output,torch.zeros(5,160))
                output.sum().backward()
                self.assertGreater(float(branch.fusion.weight.grad.abs().sum()),0.)
        rows,epochs,batch=tri_clean95.study({'scratch':True,'groupfusion_study':True})
        self.assertEqual((epochs,batch),(112,16))
        self.assertEqual({r['group_fusion'] for r in rows.values()},{'silu','radial'})

    def test_early_rank_weights_restore_source_row_order_per_receiver(self):
        from experiments.pamf_lora_main.data import packet_indices, source_role_ordinals
        signal=[]; labels=[]; receivers=[]; expected=[]
        for r in range(5):
            inverse=np.argsort(packet_indices(8000,r))
            for tx in (5,6,7):
                ordinals=source_role_ordinals(tx,800)['L_s']
                signal.extend(inverse[tx*800+ordinals])
                labels.extend([tx]*len(ordinals))
                receivers.extend([r]*len(ordinals))
                expected.extend(np.where((tx in (6,7))&(ordinals<160),3.,1.))
        actual=tri_clean95.early_source_weights(np.array(labels),np.array(receivers),np.array(signal),3.)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(int((actual==3).sum()),105)
        rows,epochs,batch=tri_clean95.study({'scratch':True,'rankweight_study':True})
        self.assertEqual((epochs,batch),(112,16))
        self.assertEqual(rows['phase_rankweight']['early_rank_weight'],3.)

    def test_confusable_pair_margin_uses_only_the_original_labeled_logits(self):
        logits=torch.zeros(5,10,requires_grad=True)
        target=torch.tensor([1,3,7,6,9])
        rx=torch.zeros_like(target)
        reference=clean_source_ce(logits,target,rx)
        constrained=clean_source_ce(logits,target,rx,pair_margin_weight=.25)
        torch.testing.assert_close(constrained-reference,.25*torch.nn.functional.softplus(torch.tensor(1.5)))
        (constrained-reference).backward()
        torch.testing.assert_close(logits.grad.nonzero(),torch.tensor([[0,1],[0,4],[1,2],[1,3],[2,7],[2,8]]))
        rows,epochs,batch=tri_clean95.study({'scratch':True,'pairmargin_study':True})
        self.assertEqual((epochs,batch),(112,16))
        self.assertEqual(rows['phase_pairmargin']['pair_margin_weight'],.25)

    def test_rtl_weight_applies_only_to_the_labeled_source_receiver(self):
        rows,epochs,u_batch=tri_clean95.study({'scratch':True,'rtl_study':True})
        self.assertEqual((epochs,u_batch),(112,16))
        self.assertEqual(rows['phase_rtl']['rtl_weight'],2.)
        t=torch.zeros(3,10)
        t[1,1]=5.
        labels=torch.zeros(3,dtype=torch.long)
        rx=torch.tensor([0,SOURCE.index('rtl_1'),2])
        ce=torch.nn.functional.cross_entropy(t,labels,label_smoothing=.01,reduction='none')
        torch.testing.assert_close(clean_source_ce(t,labels,rx),ce.mean())
        torch.testing.assert_close(clean_source_ce(t,labels,rx,2.),(ce.sum()+ce[1])/4)
