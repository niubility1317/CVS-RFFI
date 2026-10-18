"""Fourteen actual hook metrics; original raw RD remains in shape_steps."""
CONCAT_FIELDS=('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')
NO_U_FIELDS=('U_packets','U_student_NN','teacher_train_NN','teacher_eval_NN','EMA_updates','L_forwards','total_L_scale')
INFO_KEYS=tuple('concat/'+k for k in CONCAT_FIELDS)+tuple('no_u/'+k for k in NO_U_FIELDS)
