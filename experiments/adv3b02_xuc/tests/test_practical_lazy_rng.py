"""Lazy named RNGs preserve independent streams and exact channel execution."""
from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from leo_practical import Config, SCENARIOS, apply_leo_practical_channel_batch
from leo_practical.channel import ChannelStream, receiver_for_session, stable_seed
from leo_practical.execution import Execution, FAST, REFERENCE
from leo_practical.named_rng import NamedRNGs, STREAM_NAMES


def test_named_rng_only_constructs_requested_streams(monkeypatch):
    original = np.random.default_rng
    calls = []
    def counted(seed):
        calls.append(seed)
        return original(seed)
    monkeypatch.setattr(np.random, 'default_rng', counted)
    streams = NamedRNGs(192, stable_seed)
    assert list(streams) == list(STREAM_NAMES) and len(streams) == 9
    assert calls == []
    assert 'equalizer' in streams and calls == []
    noise = streams['noise']
    assert streams['noise'] is noise and len(calls) == 1
    assert np.array_equal(noise.normal(size=20), original(stable_seed(192, 'noise')).normal(size=20))
    with pytest.raises(KeyError):
        streams['not_a_stream']
    assert len(calls) == 1
    copied = deepcopy(streams)
    assert copied['noise'].bit_generator.state == streams['noise'].bit_generator.state
    assert copied['noise'] is not streams['noise']
    assert {key for key, _ in streams.items()} == set(STREAM_NAMES)
    assert len(calls) == 9


@pytest.mark.parametrize('scene', list(SCENARIOS))
@pytest.mark.parametrize('route,eq,method', [
    ('full', False, 'zf'), ('full', True, 'zf'),
    ('full', True, 'mmse'), ('residual', False, 'zf'),
])
def test_lazy_channel_iq_complete_metadata_and_terminal_state(scene, route, eq, method):
    cfg = Config(fs_hz=25e6, fc_hz=2.462e9, scenario=scene,
                 processing_route=route, equalization_enabled=eq, equalizer_method=method)
    x = np.random.default_rng(41).normal(size=(3, 2, 256))
    kwargs = dict(seed=392005, sample_ids=['a', 'b', 'c'], session_ids=['rx0', 'rx1', 'rx0'],
                  realization_namespace='source_dynamic_E82_U')
    old, metadata, states = apply_leo_practical_channel_batch(x, cfg, **kwargs)
    new, lazy_metadata, lazy_states = apply_leo_practical_channel_batch(
        x, cfg, execution=replace(REFERENCE, lazy_rng=True), **kwargs)
    assert np.array_equal(old, new)
    assert metadata == lazy_metadata
    assert np.array_equal(states, lazy_states)


@pytest.mark.parametrize('eq,method', [(False, 'zf'), (True, 'zf'), (True, 'mmse')])
def test_continuous_stream_and_all_generator_states_match(eq, method):
    cfg = Config(fs_hz=25e6, fc_hz=2.462e9, scenario='practical_low_urban',
                 equalization_enabled=eq, equalizer_method=method)
    receiver = receiver_for_session(cfg, 20, 'rx1')
    old = ChannelStream(cfg, 31, receiver)
    lazy = ChannelStream(cfg, 31, receiver, execution=replace(REFERENCE, lazy_rng=True))
    assert 'noise' not in lazy.rngs.materialized_names
    assert 'equalizer' not in lazy.rngs.materialized_names
    x = np.random.default_rng(55).normal(size=(2, 512))
    for chunk in (x[:, :256], x[:, 256:]):
        a, ma = old.process(chunk[0] + 1j * chunk[1], quality_snr_db=35.)
        b, mb = lazy.process(chunk[0] + 1j * chunk[1], quality_snr_db=35.)
        assert np.array_equal(a, b) and ma == mb
    if eq:
        assert 'equalizer' in lazy.rngs.materialized_names
    else:
        assert 'equalizer' not in lazy.rngs.materialized_names
        assert len(lazy.rngs.materialized_names) == 8
    for name in STREAM_NAMES:
        assert old.rngs[name].bit_generator.state == lazy.rngs[name].bit_generator.state


def test_supplied_geometry_does_not_construct_unused_geometry_rng():
    cfg = Config(fs_hz=25e6, fc_hz=2.462e9, scenario='practical_mid')
    receiver = receiver_for_session(cfg, 9, 'rx')
    reference = ChannelStream(cfg, 17, receiver)
    lazy = ChannelStream(cfg, 17, receiver, geometry=reference.geometry,
                         execution=replace(REFERENCE, lazy_rng=True))
    assert 'geometry' not in lazy.rngs.materialized_names
    assert list(lazy.rngs) == list(STREAM_NAMES)


def test_default_and_fast_eager_with_explicit_lazy_control():
    assert not Execution().lazy_rng and not REFERENCE.lazy_rng and not FAST.lazy_rng
    cfg = Config(fs_hz=25e6, fc_hz=2.462e9, scenario='practical_mid')
    receiver = receiver_for_session(cfg, 9, 'rx')
    eager = ChannelStream(cfg, 17, receiver)
    fast = ChannelStream(cfg, 17, receiver, execution=FAST)
    experimental = ChannelStream(cfg, 17, receiver, execution=replace(FAST, lazy_rng=True))
    assert type(eager.rngs) is dict and len(eager.rngs) == 9
    assert type(fast.rngs) is dict and len(fast.rngs) == 9
    assert isinstance(experimental.rngs, NamedRNGs)
    assert len(experimental.rngs.materialized_names) == 7


def test_pickle_preserves_accessed_and_unaccessed_streams():
    original = NamedRNGs(283, stable_seed)
    original['phase'].normal(size=19)
    restored = pickle.loads(pickle.dumps(original))
    assert restored.materialized_names == ('phase',)
    assert restored['phase'] is not original['phase']
    for name in STREAM_NAMES:
        assert np.array_equal(original[name].normal(size=17), restored[name].normal(size=17))


@pytest.mark.parametrize('route,eq,method', [
    ('full', False, 'zf'), ('full', True, 'zf'),
    ('full', True, 'mmse'), ('residual', False, 'zf'),
])
def test_spawn_workers_with_lazy_execution_match_reference(route, eq, method):
    from cvsrffi.practical_parallel import parallel_batch, shutdown
    cfg = Config(fs_hz=25e6, fc_hz=2.462e9, scenario='practical_low_urban',
                 processing_route=route, equalization_enabled=eq, equalizer_method=method)
    x = np.random.default_rng(91).normal(size=(4, 2, 256))
    kwargs = dict(seed=392005, sample_ids=['a', 'b', 'c', 'd'], session_ids=['rx'] * 4,
                  realization_namespace='source_dynamic_E121_L', return_meta=True)
    old, metadata, states = apply_leo_practical_channel_batch(x, cfg, **kwargs)
    try:
        new, lazy_metadata, lazy_states = parallel_batch(
            x, cfg, workers=2, execution=replace(REFERENCE, lazy_rng=True), **kwargs)
    finally:
        shutdown()
    assert np.array_equal(old, new) and metadata == lazy_metadata
    assert np.array_equal(states, lazy_states)
