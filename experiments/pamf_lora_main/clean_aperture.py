"""Convert native-rate LoRa packet arrays to the registered clean IQ apertures."""
import numpy as np
from scipy.signal import resample_poly

from .data import WINDOW_START
from .packet_data import CHIRPS, PACKET_LEN


ALL_CHIRPS = tuple(range(PACKET_LEN // 1024))


def padded_crop(chirps,offset=0):
    starts=np.asarray(chirps)*1024+WINDOW_START+offset-125
    crop=starts[:,None]+np.arange(1250)
    # Reflect filter support outside the recorded preamble. All central
    # 1 ms signal windows still consist of the original physical samples.
    crop=np.where(crop<0,-crop,crop)
    return np.where(crop>=PACKET_LEN,2*PACKET_LEN-2-crop,crop)


CROPS = {chirps: padded_crop(chirps) for chirps in (CHIRPS, ALL_CHIRPS)}


def clean_iq(packets,samples,chirps=CHIRPS,*,offset=0):
    if samples not in (192,256,384,512):
        raise ValueError('Unknown registered clean aperture')
    if not -WINDOW_START<=offset<=1024-1000-WINDOW_START:
        raise ValueError('Clean aperture extends outside the recorded chirp')
    crop=CROPS[chirps] if offset==0 else padded_crop(chirps,offset)
    z = resample_poly(packets[:,crop],up=samples//8,down=125,axis=-1)
    edge=samples//8
    z=z[...,edge:edge+samples]
    norm=np.sqrt(np.mean(np.abs(z)**2,axis=(1,2),keepdims=True))
    if not np.isfinite(norm).all() or (norm<=1e-12).any():
        raise ValueError('Nonfinite or empty clean packet')
    z=z/norm
    return np.stack((z.real,z.imag),axis=2).astype(np.float32,copy=False)
