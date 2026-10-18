CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
RISK_FIELDS=('selected_packets','selected_CE_old','selected_CE_new','mixed_CE_delta','loss_delta','extra_NN','weight_min','weight_max','weight_sum')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+tuple('hardrisk/'+k for k in RISK_FIELDS)
