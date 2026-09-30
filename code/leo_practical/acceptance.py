"""Synthetic numerical acceptance; no real dataset, labels or training used."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from . import (Config, ChannelStream, Receiver, receiver_for_session, Geometry,
               apply_leo_practical_channel_batch, iq_imbalance, compensate_iq,
               state_probabilities, transition_matrix)
from .channel import slant_range, fractional_delay_kernel, receiver_quality


def run():
    results = []
    def check(name, fn):
        start = time.perf_counter()
        try:
            detail = fn()
            results.append(dict(name=name, status="PASS", detail=detail,
                                seconds=time.perf_counter()-start))
        except Exception as exc:
            results.append(dict(name=name, status="FAIL", error=repr(exc),
                                seconds=time.perf_counter()-start))
    rng = np.random.default_rng(481)
    x = (rng.normal(size=(6, 256))+1j*rng.normal(size=(6, 256))).astype(np.complex64)
    cfg = Config(fs_hz=25e6)  # Synthetic test rate, NOT inferred dataset metadata.
    ids = [f"synthetic-{i}" for i in range(6)]
    sessions = ["rx-a"]*3+["rx-b"]*3
    def batch(arr, names, rx, config=cfg, namespace="acceptance"):
        return apply_leo_practical_channel_batch(arr, config, seed=52,
            sample_ids=names, session_ids=rx, realization_namespace=namespace,
            receiver_seed=81)

    def basic():
        y, meta, state = batch(x, ids, sessions)
        assert y.shape == x.shape and np.isfinite(y).all() and state.shape == (6,)
        assert all(m["frequency_compensation_applied"] and m["iq_compensation_applied"]
                   and (m["phase_tracking_applied"] == (m["receiver_lock_status"] != "unlocked")) for m in meta)
        json.dumps(meta, allow_nan=False)
        return {"output_shape": list(y.shape), "default_compensation_flags": True}
    check("default_runtime_and_metadata", basic)

    def replay():
        y, _, _ = batch(x, ids, sessions)
        order = [5, 2, 0, 4, 1, 3]
        perm, _, _ = batch(x[order], [ids[i] for i in order], [sessions[i] for i in order])
        parts = [batch(x[i:i+1], ids[i:i+1], sessions[i:i+1])[0] for i in range(6)]
        assert np.array_equal(perm, y[order]) and np.array_equal(np.concatenate(parts), y)
        different, _, _ = batch(x, ids, sessions, namespace="acceptance-other")
        assert not np.array_equal(y, different)
        return "exact replay across order/batch partition; namespace separates draws"
    check("snapshot_reproducibility", replay)

    def hardware():
        _, meta, _ = batch(x, ids, sessions)
        assert meta[0]["receiver"] == meta[2]["receiver"]
        assert meta[0]["receiver"] != meta[3]["receiver"]
        alternate = receiver_for_session(replace(cfg, scenario="practical_high"), 81, "rx-a")
        assert alternate == receiver_for_session(cfg, 81, "rx-a")
        return "hardware stable per session and across scenes"
    check("receiver_session_persistence", hardware)

    def iq():
        rx = Receiver("calibration", 0., 6., 0., 0., 6.)
        damaged = iq_imbalance(x[0].astype(np.complex128), rx)
        restored = compensate_iq(damaged, 0., 6.)
        error = float(np.max(abs(restored-x[0])))
        assert error < 1e-12
        positive_tone = np.exp(2j*np.pi*17*np.arange(256)/256)
        spectrum = np.fft.fft(iq_imbalance(positive_tone, rx))
        assert abs(spectrum[-17]) > 1
        return {"inverse_max_error": error, "pure_phase_image_amplitude": float(abs(spectrum[-17]))}
    check("iq_pure_phase_image_and_inverse", iq)

    def frequency():
        base = replace(cfg, fs_hz=1e6, rms_delay_ns=(0.,0.,0.), los_std_db=(0.,0.,0.),
            diffuse_power_db=(-300.,-300.,-300.), phase_noise_linewidth_hz=0.,
            phase_tracking_enabled=False, agc_enabled=False, state_mix_time_s=1e12,
            residual_cfo_std_hz=100., iq_amp_max_db=0., iq_phase_max_deg=0.,
            iq_estimation_amp_std_db=0., iq_estimation_phase_std_deg=0., quality_adaptation_enabled=False)
        g = Geometry(600000., 30., float(slant_range(600000.,30.)), 5000., "synthetic_control")
        measured = {}
        for mode in ("pre_sync", "post_sync"):
            c = replace(base, mode=mode)
            stream = ChannelStream(c, 14, receiver_for_session(c, 3, "rx"), g)
            stream.state = 0
            _, meta, trace = stream.process(np.ones(1024, dtype=complex), return_traces=True)
            signal = trace["pre_agc_signal"]
            estimate = float(np.mean(np.angle(signal[1:]*signal[:-1].conj()))*c.fs_hz/(2*np.pi))
            assert abs(estimate-meta["output_frequency_hz"]) < 1e-6
            measured[mode] = estimate
        assert abs(measured["pre_sync"]) > 10000 and abs(measured["post_sync"]) < 500
        return measured
    check("measured_frequency_compensation", frequency)

    def phase_tracking():
        base = replace(cfg, fs_hz=1e6, rms_delay_ns=(0.,0.,0.), los_std_db=(0.,0.,0.),
            diffuse_power_db=(-300.,-300.,-300.), phase_noise_linewidth_hz=100.,
            phase_tracking_residual_std_deg=0., residual_cfo_std_hz=0.,
            agc_enabled=False, state_mix_time_s=1e12, iq_amp_max_db=0., iq_phase_max_deg=0.,
            iq_estimation_amp_std_db=0., iq_estimation_phase_std_deg=0., quality_adaptation_enabled=False)
        values = {}
        for enabled in (False, True):
            c = replace(base, phase_tracking_enabled=enabled)
            stream = ChannelStream(c, 14, receiver_for_session(c, 3, "rx"))
            stream.state = 0
            _, _, trace = stream.process(np.ones(2048, dtype=complex), return_traces=True)
            signal = trace["pre_agc_signal"]
            values[str(enabled)] = float(np.std(np.angle(signal[1:]*signal[:-1].conj())))
        assert values["False"] > .01 and values["True"] < 1e-10
        return {"phase_increment_std_rad":values,"scope":"zero_error_proxy_control_not_PLL_accuracy"}
    check("phase_tracking_actually_changes_waveform", phase_tracking)

    def context_guard():
        c = replace(cfg, boundary="require_context")
        stream = ChannelStream(c, 7, receiver_for_session(c, 3, "rx"))
        try:
            stream.process(x[0])
        except ValueError as exc:
            assert "preceding" in str(exc)
        else:
            raise AssertionError("missing real context was accepted")
        y, meta = stream.process(x[0], preceding_context=np.ones(stream.history_size, dtype=complex))
        assert np.isfinite(y).all() and meta["boundary"] == "provided_context"
        return "missing-context refusal and explicit-context execution passed"
    check("real_context_boundary_contract", context_guard)

    def noise_agc():
        rows = []
        for loss in (0.,20.):
            c = replace(cfg, atmosphere_loss_db=loss, atmosphere_source="synthetic_attenuation_control")
            stream = ChannelStream(c, 22, receiver_for_session(c, 3, "rx"))
            y, meta = stream.process(x[0])
            rows.append(meta)
            assert abs(np.sqrt(np.mean(abs(y)**2))-1) < 1e-6
        delta = rows[1]["signal_to_added_noise_db"]-rows[0]["signal_to_added_noise_db"]
        assert abs(delta+20) < 1e-8
        assert rows[0]["measured_added_noise_power_pre_agc"] == rows[1]["measured_added_noise_power_pre_agc"]
        return {"added_attenuation_db":20., "measured_snr_delta_db":delta, "noise_floor_unchanged":True}
    check("fixed_noise_floor_and_agc", noise_agc)

    def streaming():
        c = replace(cfg, agc_enabled=False, quality_adaptation_enabled=False)
        hardware = receiver_for_session(c, 3, "rx")
        whole = ChannelStream(c, 33, hardware).process(x[0])[0]
        stream = ChannelStream(c, 33, hardware)
        split = np.concatenate([stream.process(x[0,:128])[0], stream.process(x[0,128:])[0]])
        error = float(np.max(abs(whole-split)))
        assert error < 2e-6
        return {"continuous_block_partition_max_error":error, "agc_disabled_for_comparison":True}
    check("continuous_state_and_delay_history", streaming)

    def markov():
        for environment in ("urban", "suburban"):
            p = state_probabilities(20., environment)
            transition = transition_matrix(p, 0.17, 1.)
            np.testing.assert_allclose(p@transition, p, atol=1e-14)
            np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-14)
        return "stationary mapping and stochastic rows verified analytically/numerically"
    check("markov_stationary_distribution", markov)

    def pdp():
        stream = ChannelStream(cfg, 6, receiver_for_session(cfg, 4, "rx"))
        values = []
        for state in range(3):
            weights = stream.diffuse[state]*stream.pdp.copy()
            weights[0] += stream.mean_los_power[state]
            weights /= weights.sum()
            delay = stream.delays[state]/cfg.fs_hz
            value = float(np.sqrt(weights@((delay-weights@delay)**2))*1e9)
            assert abs(value-cfg.rms_delay_ns[state]) < 1e-9
            values.append(value)
        return {"ensemble_rms_delays_ns":values}
    check("total_pdp_delay_and_power_convention", pdp)

    def torch_adapter():
        import torch
        real = np.stack([x.real,x.imag],axis=1)
        native = batch(real, ids, sessions)[0]
        tensor = torch.tensor(real, requires_grad=True)
        converted, _, state = batch(tensor, ids, sessions)
        assert converted.shape == tensor.shape and not converted.requires_grad
        np.testing.assert_array_equal(converted.numpy(), native)
        return {"torch_version":torch.__version__, "device":str(converted.device), "state_dtype":str(state.dtype)}
    check("torch_cpu_adapter", torch_adapter)

    def external():
        def callback(y, context):
            raise AssertionError("callback must not execute")
        try:
            ChannelStream(cfg, 4, receiver_for_session(cfg, 1, "rx"), receiver_processor=callback)
        except ValueError as exc:
            assert "RFF-safe" in str(exc)
        else:
            raise AssertionError("arbitrary receiver transform accepted")
        return "external equalizers/transforms rejected before execution"
    check("external_equalization_forbidden", external)

    def fractional_response():
        fraction = 0.5
        weights = fractional_delay_kernel(fraction, cfg.fractional_delay_half_length)
        frequencies = np.linspace(0, .5, 1001)
        response = np.exp(-2j*np.pi*frequencies[:,None]*np.arange(len(weights)))@weights
        band = frequencies <= .4
        max_db = float(np.max(abs(20*np.log10(np.maximum(abs(response[band]),1e-12)))))
        assert max_db < .01
        return {"max_magnitude_deviation_to_0_4fs_db":max_db,
                "common_latency_samples":cfg.fractional_delay_half_length}
    check("fractional_delay_frequency_response", fractional_response)

    def quality():
        states = [receiver_quality(cfg, snr) for snr in (25.,10.,0.)]
        assert [s[0] for s in states] == ["locked","degraded","unlocked"]
        assert states[0][1] < states[1][1] < states[2][1]
        rows = []
        for snr in (25.,10.,0.):
            stream = ChannelStream(cfg, 13, receiver_for_session(cfg, 7, "rx"))
            _, m = stream.process(x[0], quality_snr_db=snr)
            rows.append(m)
        assert rows[0]["phase_tracking_applied"] and rows[1]["phase_tracking_applied"]
        assert not rows[2]["phase_tracking_applied"] and not rows[2]["fine_frequency_tracking_applied"]
        assert rows[2]["coarse_doppler_compensation_applied"]
        assert all(not r["channel_equalization_applied"] for r in rows)
        return {"states":states,"unlocked_keeps_coarse_only":True}
    check("quality_conditioned_lock_and_compensation", quality)

    def rff_preservation():
        c = replace(cfg, rms_delay_ns=(0.,0.,0.), los_std_db=(0.,0.,0.),
            diffuse_power_db=(-300.,-300.,-300.), reference_snr_db=200.,
            phase_noise_linewidth_hz=0., residual_cfo_std_hz=0.,
            phase_tracking_residual_std_deg=0., iq_estimation_amp_std_db=0.,
            iq_estimation_phase_std_deg=0., agc_enabled=False, state_mix_time_s=1e12)
        carrier = np.exp(2j*np.pi*.071*np.arange(512)) + .3*np.exp(2j*np.pi*.13*np.arange(512))
        tx = (carrier + .08*carrier.conj()) * np.exp(.2j*abs(carrier)**2)
        tx += .04*abs(carrier)**2*carrier
        tx *= np.exp(2j*np.pi*.001*np.arange(512))
        geometry = Geometry(600000.,60.,float(slant_range(600000.,60.)),0.,"synthetic_identity_propagation")
        stream = ChannelStream(c, 19, receiver_for_session(c, 2, "rx"), geometry)
        stream.state = 0
        y,m = stream.process(tx[128:384], preceding_context=tx[:128])
        latency = c.fractional_delay_half_length
        expected = tx[128-latency:384-latency]
        error = float(np.max(abs(y-expected)))
        assert error < 3e-7
        assert not m["tx_impairment_correction_applied"]
        return {"tx_iq_pa_and_cfo_preserved_max_error":error,"known_filter_latency_accounted":True}
    check("injected_RX_correction_preserves_input_TX_RFF", rff_preservation)

    def residual_skip():
        from .residual import ResidualChannel
        original = ChannelStream.process
        def forbidden(*args, **kwargs):
            raise AssertionError("full waveform path must not execute")
        ChannelStream.process = forbidden
        try:
            c = replace(cfg, processing_route="residual")
            result,meta,_ = batch(x,ids,sessions,config=c)
            assert np.isfinite(result).all()
            assert all(not m["full_channel_waveform_synthesized"] for m in meta)
            assert all(not m["channel_equalization_applied"] for m in meta)
        finally:
            ChannelStream.process = original
        return "residual route works with full process forcibly unavailable; EQ defaults off"
    check("residual_does_not_synthesize_full_channel", residual_skip)

    def route_equivalence():
        from .residual import ResidualChannel
        base = replace(cfg, quality_adaptation_enabled=False, phase_noise_linewidth_hz=0.,
            phase_tracking_residual_std_deg=0., residual_cfo_std_hz=0.,
            iq_amp_max_db=0.,iq_phase_max_deg=0., iq_estimation_amp_std_db=0.,
            iq_estimation_phase_std_deg=0.,los_std_db=(0.,0.,0.),
            scatter_doppler_hz=0.,agc_enabled=False,state_mix_time_s=1e12)
        geometry = Geometry(600000.,60.,float(slant_range(600000.,60.)),0.,"frozen_equivalence_control")
        local = np.random.default_rng(55)
        signal = local.normal(size=2048)+1j*local.normal(size=2048)
        errors = {}
        for enabled in (False,True):
            c = replace(base,equalization_enabled=enabled)
            receiver = receiver_for_session(c,9,"rx")
            full = ChannelStream(c,35,receiver,geometry)
            direct = ResidualChannel(replace(c,processing_route="residual"),35,receiver,geometry)
            y,m = full.process(signal)
            r,mr = direct.process(signal)
            guard = full.history_size+c.equalizer_length
            error = float(np.max(abs(y[guard:]-r[guard:])))
            assert error < 5e-6
            assert m["state_start"]==mr["state_start"] and m["receiver"]==mr["receiver"]
            assert m["channel_equalization_applied"]==enabled==mr["channel_equalization_applied"]
            errors[str(enabled)] = error
        return {"frozen_channel_interior_max_error":errors,"boundary_guard_excluded":True}
    check("full_vs_direct_residual_frozen_equivalence", route_equivalence)

    def equalizers():
        from .equalization import design_equalizer,linear_filter
        local = np.random.default_rng(89)
        signal = local.normal(size=4096)+1j*local.normal(size=4096)
        h = np.array([1.,.6*np.exp(.2j),.2j])
        damaged,_=linear_filter(signal,h)
        values = {}
        for method in ("mmse","zf"):
            c=replace(cfg,equalization_enabled=True,equalizer_method=method,channel_estimation_nmse_db=-300.)
            g,m=design_equalizer(h,c,np.random.default_rng(1),1e-6,1.,1.)
            out,_=linear_filter(damaged,g)
            guard=200
            expected=signal[guard-c.equalizer_delay_samples:len(signal)-c.equalizer_delay_samples]
            mse=float(np.mean(abs(out[guard:]-expected)**2))
            assert mse < 1e-5
            assert np.max(abs(np.fft.fft(g,4096))) <= 10**(c.equalizer_max_gain_db/20)+.01
            values[method]=mse
        return {"known_channel_ISI_recovery_mse":values,"not_TX_distortion_correction":True}
    check("optional_mmse_and_regularized_zf", equalizers)

    def residual_replay():
        c=replace(cfg,processing_route="residual",equalization_enabled=True)
        y,m,_=batch(x,ids,sessions,config=c)
        order=[5,3,1,4,2,0]
        other,_,_=batch(x[order],[ids[i] for i in order],[sessions[i] for i in order],config=c)
        np.testing.assert_array_equal(y[order],other)
        json.dumps(m,allow_nan=False)
        return "residual plus EQ reproducible and metadata serializable"
    check("residual_equalization_replay", residual_replay)

    def six_scenarios():
        # Exercise actual templates through both receivers and EQ branches;
        # fixed quality ensures the requested equalizer really executes.
        from .channel import SCENARIOS
        expected = {
            "practical_high": ("suburban", (45., 80.)),
            "practical_mid": ("suburban", (20., 45.)),
            "practical_low_urban": ("urban", (10., 30.)),
            "practical_low_suburban": ("suburban", (10., 30.)),
            "practical_mid_urban": ("urban", (20., 45.)),
            "practical_high_urban": ("urban", (45., 80.)),
        }
        assert SCENARIOS == expected
        exercised = []
        for name, (environment, (lo, hi)) in expected.items():
            fields = json.loads((Path(__file__).parent / "configs" / (name+".json")).read_text(encoding="utf-8"))
            assert fields["fs_hz"] is None and fields["scenario"] == name
            base = Config(**{**fields, "fs_hz": cfg.fs_hz})
            assert not base.equalization_enabled and base.processing_route == "full"
            for route in ("full", "residual"):
                for eq in ("off", "mmse", "zf"):
                    c = replace(base, processing_route=route, equalization_enabled=eq!="off",
                                equalizer_method="mmse" if eq=="off" else eq,
                                quality_adaptation_enabled=False)
                    y, meta, _ = batch(x[:2], ids[:2], sessions[:2], config=c)
                    assert y.shape == x[:2].shape and np.isfinite(y).all()
                    again, _, _ = batch(x[1::-1], ids[1::-1], sessions[1::-1], config=c)
                    np.testing.assert_array_equal(y[::-1], again)
                    for m in meta:
                        elevation = m["geometry"]["elevation_deg"]
                        assert lo <= elevation <= hi
                        assert m["scenario"] == name and m["processing_route"] == route
                        assert m["channel_equalization_applied"] == (eq!="off")
                        assert m["full_channel_waveform_synthesized"] == (route=="full")
                        np.testing.assert_allclose(m["state_probabilities"], state_probabilities(elevation, environment))
                    json.dumps(meta, allow_nan=False)
                    exercised.append([name, route, eq])
        return {"config_route_equalizer_combinations": len(exercised), "combinations": exercised,
                "quality_adaptation_disabled_only_for_branch_coverage": True}
    check("six_scene_templates_routes_equalizers_replay", six_scenarios)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = run()
    report = dict(interpreter=sys.executable, numpy_version=np.__version__,
        scope="synthetic_numerical_checks_no_training_no_real_data",
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        checks=results, passed=sum(r["status"]=="PASS" for r in results),
        failed=sum(r["status"]=="FAIL" for r in results))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(bool(report["failed"]))
