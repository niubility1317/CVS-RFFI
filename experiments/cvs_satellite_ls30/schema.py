"""Only metrics emitted by this hook;full6L total and no invented shape/RD."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
SATLS_FIELDS=('selected_packets','epsilon','selected_CE_old','selected_CE_new','mixed_CE_delta','loss_delta','extra_NN')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+tuple('satls/'+k for k in SATLS_FIELDS)
