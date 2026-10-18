"""Actual emitted scalars;concat total includes all original6L."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
BANK_FIELDS=('bank_weight_sum','bin0','bin1','bin2','bin3','sampling_cpu_wall_seconds','weighted_loss_cpu_wall_seconds','extra_NN')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+tuple('bank/'+k for k in BANK_FIELDS)
