"""Only actual returned native metrics; geometry is stored separately."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+('shape/RD','split/As_RD')
