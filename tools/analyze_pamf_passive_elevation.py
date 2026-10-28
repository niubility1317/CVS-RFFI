"""Deterministic geometry analysis for single-satellite passive RFFI.

No IQ, target labels, learned model, real ephemeris, or network is accessed.
Examples are explicit circular-orbit reference cases, never mission estimates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


R = 6371000.
H = 600000.
GM = 3.986004418e14
SIDEREAL_DAY_S = 86164.0905


def psi(elevation_deg, altitude_m=H):
    e = math.radians(elevation_deg)
    return math.acos(R/(R+altitude_m)*math.cos(e))-e


def area_mass(lo, hi, minimum=10., maximum=80.):
    a,b=max(lo,minimum),min(hi,maximum)
    if a>=b:return 0.
    return ((math.cos(psi(b))-math.cos(psi(a))) /
            (math.cos(psi(maximum))-math.cos(psi(minimum))))


def pass_half_angle(elevation, maximum_elevation):
    if elevation>=maximum_elevation:return 0.
    ratio=math.cos(psi(elevation))/math.cos(psi(maximum_elevation))
    return math.acos(min(1.,max(-1.,ratio)))


def pass_time_profile(maximum_elevation, minimum=10., upper=80.):
    if not minimum<min(maximum_elevation,upper):raise ValueError('Empty pass window')
    a=lambda e:pass_half_angle(e,maximum_elevation)
    den=a(minimum)-a(upper)
    masses=[]
    for lo,hi in ((10.,30.),(30.,45.),(45.,upper)):
        lo,hi=max(lo,minimum),min(hi,upper,maximum_elevation)
        masses.append(0. if lo>=hi else (a(lo)-a(hi))/den)
    if not math.isclose(sum(masses),1.,abs_tol=1e-10):raise AssertionError(masses)
    return dict(peak_elevation_deg=maximum_elevation,minimum_elevation_deg=minimum,
                upper_elevation_deg=upper,low_mid_high=masses,
                assumptions='one complete circular pass, fixed ground point, no Earth rotation within pass; uniform time and constant emission')


def cone_minimum_elevation(half_angle):
    value=(R+H)/R*math.sin(math.radians(half_angle))
    return 0. if value>=1 else math.degrees(math.acos(value))


def fractions(elevation, lower=10., upper=80.):
    keep=(elevation>=lower)&(elevation<upper)
    n=int(keep.sum())
    if n==0:return None
    return dict(accepted_samples=n,
                low_mid_high=[float(np.count_nonzero(keep&(elevation>=a)&(elevation<b))/n)
                              for a,b in ((10.,30.),(30.,45.),(45.,upper))])


def orbit_elevations(inclination, latitude, *, days=14, step_s=5.):
    # RAAN0, initial anomaly0, ground longitude0. These are reference choices.
    t=np.arange(0.,days*86400.,step_s,dtype=np.float64)
    rs=R+H;inc=math.radians(inclination);lat=math.radians(latitude)
    u=math.sqrt(GM/rs**3)*t
    earth=2*math.pi/SIDEREAL_DAY_S*t
    dot=rs*(math.cos(lat)*(np.cos(u)*np.cos(earth)+math.cos(inc)*np.sin(u)*np.sin(earth))
            +math.sin(inc)*np.sin(u)*math.sin(lat))
    distance=np.sqrt(rs**2+R**2-2*R*dot)
    return np.rad2deg(np.arcsin(np.clip((dot-R)/distance,-1.,1.)))


def analyze():
    cases=[]
    for inclination,latitude in ((53.,0.),(53.,30.),(53.,50.),(53.,60.),(97.6,0.),(97.6,30.),(97.6,50.)):
        e=orbit_elevations(inclination,latitude)
        v90=fractions(e,10.,90.)
        baseline=fractions(e)
        masks=[]
        for lower in (10.,20.,25.,30.,40.):
            f=fractions(e,lower,80.)
            masks.append(dict(minimum_ground_elevation_deg=lower,profile=f,
                              retained_fraction_of_baseline_visible=(f['accepted_samples']/baseline['accepted_samples'] if f else 0.)))
        cones=[]
        for half in (60.,50.,45.):
            lower=max(10.,cone_minimum_elevation(half))
            f=fractions(e,lower,80.)
            cones.append(dict(nadir_hard_cone_half_angle_deg=half,
                              minimum_ground_elevation_deg=lower,profile=f,
                              retained_fraction_of_baseline_visible=(f['accepted_samples']/baseline['accepted_samples'] if f else 0.)))
        cases.append(dict(inclination_deg=inclination,ground_latitude_deg=latitude,
                          reference_10_to_80=baseline,reference_10_to_90=v90,
                          fraction_above80_in_visible=(1-baseline['accepted_samples']/v90['accepted_samples']),
                          elevation_mask_sensitivity=masks,
                          hard_cone_sensitivity=cones))
    # A single resolution check is relevant to this numerical time integration.
    coarse=cases[1]['reference_10_to_80']['low_mid_high']
    fine=fractions(orbit_elevations(53.,30.,step_s=1.))['low_mid_high']
    error=max(abs(a-b) for a,b in zip(coarse,fine))
    if error>.002:raise AssertionError('Time grid resolution is material')
    links=[]
    ref_distance=math.sqrt((R+H)**2-(R*math.cos(math.radians(60.)))**2)-R*math.sin(math.radians(60.))
    for e in (10.,20.,30.,45.,60.,80.):
        er=math.radians(e)
        distance=math.sqrt((R+H)**2-(R*math.cos(er))**2)-R*math.sin(er)
        beta=math.degrees(math.asin(R/(R+H)*math.cos(er)))
        links.append(dict(elevation_deg=e,slant_range_km=distance/1000.,
                          extra_free_space_loss_vs60_db=20*math.log10(distance/ref_distance),
                          satellite_off_nadir_deg=beta,
                          native_model_LOS_prior_urban=1-(90-e)**2/7000.,
                          native_model_LOS_prior_suburban=1-(90-e)**2/16600.))
    area_profiles=[dict(minimum_ground_elevation_deg=m,
                        low_mid_high=[area_mass(a,b,m) for a,b in ((10.,30.),(30.,45.),(45.,80.))])
                   for m in (10.,20.,25.,30.,40.)]
    pass_profiles=[pass_time_profile(peak,minimum) for peak in (30.,45.,60.,90.)
                   for minimum in (10.,25.) if minimum<peak]
    pass_profiles.append(pass_time_profile(90.,10.,90.))
    for row in area_profiles:
        if not math.isclose(sum(row['low_mid_high']),1.,abs_tol=1e-10):raise AssertionError(row)
    return dict(schema='pamf_single_satellite_passive_elevation_analysis_v1',date='2026-10-28',
                task='single satellite passive listening; no ground transmitter control or satellite selection',
                status='GEOMETRY_ANALYSIS_ONLY',experiment_config_changed=False,IQ_read=False,
                assumptions=dict(altitude_m=H,earth_radius_m=R,GM=GM,reference_days=14,
                                 time_step_s=5.,earth_rotation=True,RAAN0_deg=0.,initial_anomaly_deg=0.,ground_longitude_deg=0.,
                                 constant_packet_rate=True,detection_model=False,J2=False,
                                 measured_orbit=False,measured_antenna_pattern=False),
                bins=dict(low=[10,30],mid=[30,45],high_model_support=[45,80]),
                area_profiles=area_profiles,pass_profiles=pass_profiles,orbit_time_profiles=cases,
                link_geometry=links,resolution_check=dict(case='i53_lat30',fine_step_s=1.,max_fraction_error=error),
                interpretation='not a real mission or deployment frequency; antenna hard cones are hypothetical field-of-view sensitivity, not measured HPBW',
                sources=[dict(title='Elevation Angle Characterization for LEO Satellites: First and Second Order Statistics',
                              url='https://www.mdpi.com/2076-3417/13/7/4405'),
                         dict(title='AWS Ground Station site masks',url='https://docs.aws.amazon.com/ground-station/latest/ug/locations.site-masks.html'),
                         dict(title='ESA Near Earth Antenna Performance Requirements, issue1.2, 16March2026',
                              url='https://commercialisation.esa.int/wp-content/uploads/2025/09/ESA-DOPS-MGT-TN-0014_Antenna-Performance-Requirements_Issue1.2-20260316.pdf')])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();result=analyze()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',area=result['area_profiles'][0],
                         passes=result['pass_profiles'],
                         orbit_rows=[dict(i=r['inclination_deg'],latitude=r['ground_latitude_deg'],
                                          percent=[round(100*x,3) for x in r['reference_10_to_80']['low_mid_high']])
                                     for r in result['orbit_time_profiles']],
                         resolution_check=result['resolution_check'],experiment_config_changed=False),ensure_ascii=False))


if __name__=='__main__':main()
