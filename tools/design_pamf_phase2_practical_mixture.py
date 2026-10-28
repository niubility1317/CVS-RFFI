"""Design-only practical mixture prior. No IQ, model, target or network access.

The covered-surface assumption is explicit. It is not an ephemeris, serving-
satellite distribution, traffic measurement, or estimate from target accuracy.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE_BANDS = {
    'practical_high': ('suburban', (45., 80.)),
    'practical_mid': ('suburban', (30., 45.)),
    'practical_low_suburban': ('suburban', (10., 30.)),
    'practical_high_urban': ('urban', (45., 80.)),
    'practical_mid_urban': ('urban', (30., 45.)),
    'practical_low_urban': ('urban', (10., 30.)),
}


def model_parameters(path: Path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'RE', 'SCENARIOS', 'VERSION'}:
                    values[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.ClassDef) and node.name == 'Config':
            for field in node.body:
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name):
                    if field.target.id in {'altitude_m', 'fc_hz'}:
                        values[field.target.id] = ast.literal_eval(field.value)
    for key in ('RE', 'SCENARIOS', 'VERSION', 'altitude_m', 'fc_hz'):
        if key not in values:
            raise ValueError(f'Missing model parameter: {key}')
    for scene, (env, (lo, hi)) in SCENE_BANDS.items():
        native_env, (native_lo, native_hi) = values['SCENARIOS'][scene]
        if env != native_env or not native_lo <= lo < hi <= native_hi:
            raise ValueError(f'Conditional mixed geometry outside native support: {scene}')
    return values


def central_angle(elevation_deg, earth_radius_m, altitude_m):
    e = math.radians(elevation_deg)
    return math.acos(earth_radius_m / (earth_radius_m + altitude_m) * math.cos(e)) - e


def band_probability(lo, hi, *, earth_radius_m, altitude_m, min_elevation=10., max_elevation=80.):
    cosine = lambda e: math.cos(central_angle(e, earth_radius_m, altitude_m))
    return (cosine(hi) - cosine(lo)) / (cosine(max_elevation) - cosine(min_elevation))


def build_spec(root=ROOT):
    p = model_parameters(root / 'code/leo_practical/channel.py')
    geometry = dict(earth_radius_m=p['RE'], altitude_m=p['altitude_m'])
    bands = {name: band_probability(lo, hi, **geometry)
             for name, (lo, hi) in {'low': (10., 30.), 'mid': (30., 45.), 'high': (45., 80.)}.items()}
    if not math.isclose(sum(bands.values()), 1., abs_tol=1e-12):
        raise AssertionError('Elevation masses do not sum to one')
    scenarios = []
    for scene, (environment, limits) in SCENE_BANDS.items():
        mass = band_probability(*limits, **geometry)
        scenarios.append(dict(id=scene, environment=environment,
            probability=.5 * mass, native_elevation_range_deg=list(p['SCENARIOS'][scene][1]),
            mixed_conditional_elevation_range_deg=list(limits)))
    if not math.isclose(sum(s['probability'] for s in scenarios), 1., abs_tol=1e-12):
        raise AssertionError('Scenario masses do not sum to one')
    # Test the proposed inverse-CDF at endpoints and an interior quantile.
    for _, (_, (lo, hi)) in SCENE_BANDS.items():
        c0 = math.cos(central_angle(lo, **geometry))
        c1 = math.cos(central_angle(hi, **geometry))
        for u in (0., .5, 1.):
            psi = math.acos(c0 + u * (c1 - c0))
            # sin(e) = ((R+h)cos(psi)-R)/slant_range.
            R, H = geometry['earth_radius_m'], geometry['altitude_m']
            distance = math.sqrt((R+H)**2+R**2-2*R*(R+H)*math.cos(psi))
            elevation = math.degrees(math.asin(((R+H)*math.cos(psi)-R)/distance))
            if not lo-1e-8 <= elevation <= hi+1e-8:
                raise AssertionError('Conditional inverse-CDF outside band')
    replicates = [dict(id=f'MIX{index:02d}',
        physical_partition_index=index-1, scene_seed=2026102810+index,
        channel_seed=2026102820+index, evaluation_seed=2026102830+index)
        for index in range(1, 9)]
    return dict(schema='pamf_phase2_practical_mixture_design_v1', date='2026-10-28',
        status='DESIGN_ONLY', capsule_built=False, target_IQ_read=False,
        name='Practical residual coverage-weighted mixture',
        model_source='code/leo_practical/channel.py', channel_version=p['VERSION'],
        route=dict(processing_route='residual', mode='post_sync', equalization_enabled=False,
                   fs_hz=25e6, fc_hz=p['fc_hz'], **geometry),
        probability_basis=dict(
            geometry='uniform ground surface area in visible spherical annulus, conditioned on 10<=elevation<=80 deg',
            formula='psi(e)=acos(R/(R+h)*cos(e))-e; P(a<=e<b)=[cos(psi(b))-cos(psi(a))]/[cos(psi(80))-cos(psi(10))]',
            environment={'suburban': .5, 'urban': .5},
            environment_basis='neutral benchmark coverage prior; actual geographic/traffic frequencies unavailable',
            empirical_deployment_frequency=False, measured_ephemeris=False,
            elevation_band_probability=bands,
            overlapping_native_ranges='native mid20-45 overlaps low10-30; mixed sampler uses exclusive 10-30/30-45/45-80 bands and explicit geometry'),
        scenarios=scenarios,
        per_record_generation=[
            'choose one scenario from fixed six-category probability vector, independently of TX/class/old-new/support-query/model/method',
            'within selected band sample cos(psi) uniformly between band boundaries; derive elevation and physical Geometry; bound floating-point endpoint error to declared band',
            'keep native receiver/session construction and signed circular range-rate formula; receiver_seed fixed2027 across all sets',
            'call ResidualChannel with geometry=... once; all remaining practical fading/receiver/processing formulas unchanged',
            'store realized scene/geometry/state diagnostics on builder/scorer side, never as oracle predictor inputs'],
        generator_implementation_status='PLANNED: existing ResidualChannel accepts geometry, batch wrapper needs mixed geometry adapter',
        inner_fading_state_rule='native Nie2012 state_probabilities(elevation,environment); do not multiply or replace with outer six-scenario probabilities',
        replicates=replicates, receiver_seed=2027, physical_allocation_seed=2026102800,
        support_draw_seeds=list(range(2026092711,2026092716)),
        unique_physical_policy='all eight mixed sets and three designated single-scene diagnostic sets pairwise physical-ID disjoint; single IQ has exactly one overlay',
        single_scene_diagnostics=[
            dict(id='D-LS', scenario='practical_low_suburban', native_elevation_range_deg=[10., 30.],
                 target_scene_probability=1., dataset_count=1),
            dict(id='D-MU', scenario='practical_mid_urban', native_elevation_range_deg=[20., 45.],
                 target_scene_probability=1., dataset_count=1),
            dict(id='D-LU', scenario='practical_low_urban', native_elevation_range_deg=[10., 30.],
                 target_scene_probability=1., dataset_count=1)],
        revision='r03_three_designated_single_scene_groups',
        same_realization_for_all_methods=True, same_realization_across_model_seeds=True,
        support_pool_per_RX_class_set=36, query_per_RX_class_set=30,
        fixed_number_of_mixed_sets=8, accuracy_based_replication_extension=False,
        claims='physics-informed reference mixture over terrestrial proxy IQ; not a measured mission distribution or in-orbit validation',
        literature=[dict(title='Elevation Angle Characterization for LEO Satellites: First and Second Order Statistics',
            url='https://www.mdpi.com/2076-3417/13/7/4405',
            role='supports dependence on latitude, inclination and altitude; the covered-area reference assumption and computed weights here are our explicit design')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    spec = build_spec()
    if args.verify:
        saved = json.loads(args.verify.read_text(encoding='utf-8-sig'))
        if saved != spec:
            raise AssertionError('Saved prior differs from current model parameters and fixed design')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(spec, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', scenarios={s['id']:s['probability'] for s in spec['scenarios']},
        mixed_sets=len(spec['replicates']), IQ_read=False, experiments_started=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
