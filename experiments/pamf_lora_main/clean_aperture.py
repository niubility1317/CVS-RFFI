"""Convert native-rate LoRa packet arrays to the registered clean IQ apertures."""
import numpy as np
from scipy.signal import resample_poly

from .data import WINDOW_START
from .packet_data import CHIRPS, PACKET_LEN


ALL_CHIRPS = tuple(range(PACKET_LEN // 1024))
NATIVE_EDGE_POINTS = 12

def native_edge_positions(edge_points=NATIVE_EDGE_POINTS):
    if edge_points not in (12,16,24,32):
        raise ValueError('Unknown native chirp boundary width')
    return tuple(range(edge_points)) + tuple(range(1024-edge_points,1024))


NATIVE_EDGE_POSITIONS = native_edge_positions()


def padded_crop(chirps,offset=0):
    starts=np.asarray(chirps)*1024+WINDOW_START+offset-125
    crop=starts[:,None]+np.arange(1250)
    # Reflect filter support outside the recorded preamble. All central
    # 1 ms signal windows still consist of the original physical samples.
    crop=np.where(crop<0,-crop,crop)
    return np.where(crop>=PACKET_LEN,2*PACKET_LEN-2-crop,crop)


CROPS = {chirps: padded_crop(chirps) for chirps in (CHIRPS, ALL_CHIRPS)}


def shift_native_packets(packets,offset):
    """Interpolate a sub-sample training shift before taking any chirp views."""
    if packets.ndim!=2 or packets.shape[1]!=PACKET_LEN or not -1<offset<1:
        raise ValueError('Native timing shifts require full packets and a sub-sample offset')
    floor=int(np.floor(offset))
    phase=offset-floor
    positions=np.arange(PACKET_LEN)+floor
    weights=(-phase*(phase-1)*(phase-2)/6,
             (phase+1)*(phase-1)*(phase-2)/2,
             -(phase+1)*phase*(phase-2)/2,
             (phase+1)*phase*(phase-1)/6)
    shifted=np.zeros_like(packets)
    for tap,weight in zip((-1,0,1,2),weights,strict=True):
        idx=positions+tap
        idx=np.where(idx<0,-idx,idx)
        idx=np.where(idx>=PACKET_LEN,2*PACKET_LEN-2-idx,idx)
        shifted+=weight*packets[:,idx]
    return shifted


def clean_iq(packets,samples,chirps=CHIRPS,*,offset=0,native_edges=False,edge_points=NATIVE_EDGE_POINTS):
    if samples not in (192,256,384,512):
        raise ValueError('Unknown registered clean aperture')
    if not -WINDOW_START<=offset<=1024-1000-WINDOW_START:
        raise ValueError('Clean aperture extends outside the recorded chirp')
    if native_edges and (samples!=384 or chirps!=ALL_CHIRPS or offset):
        raise ValueError('Native boundaries require the centered 384-point, eight-chirp aperture')
    crop=CROPS[chirps] if offset==0 else padded_crop(chirps,offset)
    z = resample_poly(packets[:,crop],up=samples//8,down=125,axis=-1)
    edge=samples//8
    z=z[...,edge:edge+samples]
    norm=np.sqrt(np.mean(np.abs(z)**2,axis=(1,2),keepdims=True))
    if not np.isfinite(norm).all() or (norm<=1e-12).any():
        raise ValueError('Nonfinite or empty clean packet')
    z=z/norm
    if native_edges:
        positions=np.asarray(chirps)[:,None]*1024+np.asarray(native_edge_positions(edge_points))
        z=np.concatenate((z,packets[:,positions]/norm),axis=-1)
    return np.stack((z.real,z.imag),axis=2).astype(np.float32,copy=False)
