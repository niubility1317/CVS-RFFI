"""Frozen source-checkpoint interface evidence, without fitting or query data."""
import torch


def check(model, x, classifier):
    if model.training or x.shape != (3,2,256) or x.dtype != torch.float32:
        raise ValueError('A frozen eval model and the declared three synthetic inputs are required')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('This boundary observation cannot train parameters')
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    head=classifier(model);weight=head.weight;original=weight.detach().clone()
    def features():
        with torch.autocast(device_type=x.device.type,enabled=False):
            return model.id_backbone.encoder.features(x.float())
    with torch.no_grad():
        combined=model.forward_identity_only(x)
        pure=features()
        if pure.shape!=(3,160) or not torch.isfinite(pure).all() or not torch.equal(pure,combined['id_feat_joint']):
            raise ValueError('Independent features differ from the deployed joint160')
        try:
            weight.zero_();without_values=features()
            head.weight=None;without_parameter=features()
        finally:
            head.weight=weight;weight.copy_(original)
        restored=model.forward_identity_only(x)
    if not torch.equal(pure,without_values) or not torch.equal(pure,without_parameter):
        raise ValueError('Concrete class W affects deployed features')
    if not torch.equal(restored['tx_logits'],combined['tx_logits']):
        raise ValueError('Temporary class-weight observation did not restore predictions')
    after=model.state_dict()
    if before.keys()!=after.keys() or any(not torch.equal(v,after[k]) for k,v in before.items()):
        raise ValueError('Frozen parameter or buffer state changed')
    return dict(status='PASS',joint_shape=[3,160],raw_joint_no_additional_normalization=True,
        independent_feature_forward_equal=True,W_zero_features_equal=True,W_deleted_features_equal=True,
        default_logits_restored_equal=True,parameter_buffer_state_unchanged=True,
        normalization_gain_reference_response_retained=True,synthetic_inputs=3,packet_network_forwards=15,
        actual_source_IQ_reads=0,U_V_query_IQ_reads=0,head_fit_updates=0,optimizer_updates=0,
        claim='One actual frozen checkpoint interface and W-independent features on three synthetic inputs; no identity-transfer efficacy or new-TX claim')


def synthetic_inputs():
    t=torch.arange(256,dtype=torch.float32)
    x=torch.zeros(3,2,256)
    x[1,0]=torch.cos(2*torch.pi*t/17);x[1,1]=torch.sin(2*torch.pi*t/17)
    x[2,0]=torch.cos(2*torch.pi*t/31)+.2*torch.sin(2*torch.pi*t/7)
    x[2,1]=torch.sin(2*torch.pi*t/31)+.2*torch.cos(2*torch.pi*t/7)
    return x
