"""Only metrics actually returned by the native mixed128 hook."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
CONTRAST_FIELDS=('physical_inputs','anchors_per_term','anchors_per_physical_input','map_terms','raw_maps_used','extra_NN','true_As_anchors_per_term')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+tuple('contrast/'+k for k in CONTRAST_FIELDS)
