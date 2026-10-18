"""Fourteen metrics actually emitted;rawRD stays in original shape_steps."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
BALANCE_FIELDS=('selected_packets','scene0_packets','scene1_packets','scene2_packets','extra_scene_index','assignment_permutations','extra_NN')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+tuple('balance/'+k for k in BALANCE_FIELDS)
