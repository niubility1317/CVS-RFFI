"""Actual native metric interface; no implied or historical-only keys."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
WEIGHT_FIELDS=('mixed_weighted_RD','mixed_weighted_CE','mixed_weighted_KL','ordinary64_RD','ordinary64_CE','ordinary64_KL','selected64_RD','selected64_CE','selected64_KL','ordinary_group_weight','selected_group_weight')
RAW_FIELDS=('raw_RD','raw_CE','raw_KL')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+tuple('weight/'+k for k in WEIGHT_FIELDS+RAW_FIELDS)
