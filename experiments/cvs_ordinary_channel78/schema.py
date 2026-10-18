"""Native loss metrics plus compact primitive-bypass telemetry."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
OC_FIELDS=('ordinary_calls','ordinary_packets','MP_calls','MP_packets','AWGN_calls','AWGN_packets','shadow_MP_packets','shadow_AWGN_packets','returned_ordinary_raw_MSE','post_return_rng_preserved','extra_NN')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+tuple('ordinary_channel/'+k for k in OC_FIELDS)
