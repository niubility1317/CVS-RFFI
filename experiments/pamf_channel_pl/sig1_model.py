"""Full original PAMF with the canonical single SIG_TIME descriptor view."""


def install_single_view(model,seed):
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool
    ref=model.id_backbone.encoder
    if ref.sequence_pool.channels!=96:raise ValueError('Original PAMF t2 96 channels required')
    ref.sequence_pool=CompositePool(96,'SIG_TIME',seed)
    validate_single_view(model)
    model._feature_learning_report.update(arm='SIG_TIME',modes=['SIG_TIME'],signature_views=1,
        descriptor_width_total=112,output_embedding=160,signature_parameters=18976,
        full_time_frequency_PA_reference_retained=True)
    return model


def validate_single_view(model):
    pool=model.id_backbone.encoder.sequence_pool
    if (pool.arm!='SIG_TIME' or len(pool.views)!=1 or pool.modes!=('SIG_TIME',)
            or tuple(pool.views[0].local_projection.weight.shape)!=(6,96,1)
            or tuple(pool.output_projection.weight.shape)!=(160,112)
            or sum(p.numel() for p in pool.parameters())!=18976):raise ValueError('Exactly one learned SIG_TIME view required')
    head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
    if tuple(head.shape)!=(6,160):raise ValueError('Original six-class joint default head required')
    return pool
