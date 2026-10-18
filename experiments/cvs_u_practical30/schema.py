"""Only actual full6L scalar values;U supervision lives in u_steps,not invented L metrics."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)
