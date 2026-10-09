"""Compare the original role contract plus the separately recorded class map."""


def validate_source_contract(actual, expected, classes, dataset):
    # The original core90 role contract predates the runtime-added classes field.
    # Every original field (including physical IDs) must still match exactly.
    required_original={'role_ids','source_rxs','source_days','ratios','split_seed','num_classes',
        'checkpoint_init','target_access_before_freeze'}
    if not required_original.issubset(expected):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: incomplete original role contract')
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: '+key)
    required=dict(classes=list(classes),num_classes=len(classes),physical_roles='EXACT_MATCH',
        dataset_path=str(dataset),equalized=1,out_len=256,normalize=True)
    for key,value in required.items():
        if actual.get(key)!=value:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: '+key)
    if expected['checkpoint_init']!='scratch_only' or expected['target_access_before_freeze'] is not False:
        raise ValueError('CHECKPOINT_TARGET_CONTAMINATED')
    return dict(status='VERIFIED',original_fields_exact=list(expected),class_order_exact=True,
        physical_roles='EXACT_MATCH',representation='equalized1_center256_unitRMS',checkpoint_init='scratch_only')
