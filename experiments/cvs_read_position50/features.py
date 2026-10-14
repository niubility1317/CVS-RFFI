"""Keep the native .1 Dropout; move only the SIG_TIME descriptor hook."""
from experiments.cvs_signature_stats50.features import install as shared
def install(model,seed,arm):
 if arm not in ('SIG_TIME','SIG_TIME_PRE'):raise ValueError('Unregistered descriptor read-position arm')
 model=shared(model,seed,'SIG_TIME');ref=model.id_backbone.encoder
 ref.sequence_read_position='t2.pool' if arm=='SIG_TIME_PRE' else 't2'
 ref.sequence_pool.arm=arm
 model._feature_learning_report.update(arm=arm,layer=ref.sequence_read_position,descriptor_read_position=ref.sequence_read_position,main_body_dropout_unchanged=True,descriptor_formula_unchanged=True,parameter_matched=True)
 return model
