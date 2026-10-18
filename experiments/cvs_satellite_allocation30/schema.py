"""Metrics actually emitted;concat total covers original6L and no target-entropy delta."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
ALLOC_FIELDS=('selected_packets','active_cells','quota_min','quota_max','changed_positions','allocation_cpu_wall_seconds','extra_NN')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+tuple('allocation/'+k for k in ALLOC_FIELDS)
