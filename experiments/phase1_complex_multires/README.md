# Phase1 complex multiresolution restoration

Source-only IQ training and source IQ-only candidate selection use no transmitter labels or classifier weights. A bounded two-resolution complex spectral mask filters the frozen DURA output using received IQ aligned to that output; a fixed blind gate preserves clean signals. After model selection is frozen, the frozen Tri-SIG-TIME SSL reader evaluates the source prediction. If the source selection picks the candidate, the pipeline performs seven-view Phase1 prediction, seals all predictions, invokes the independent truth-last scorer, and compares against same-record DURA.

No model may claim independent blind test performance: prior Phase1 development results informed research. Source scripts use only L_s/U_s physical records, no U_s transmitter labels and no Phase3 data. The simulator is not validated real satellite data.
