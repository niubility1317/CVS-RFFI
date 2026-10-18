"""Fourteen actual emitted metrics;raw RD remains in original shape_steps."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
LRAW_FIELDS=('ordinary_calls','ordinary_packets','discardedordinary_MSE','discardedordinary_changed_packets','returned_raw_MSE','full_native_draws','extra_NN')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+tuple('lraw/'+k for k in LRAW_FIELDS)
