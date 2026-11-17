# POSTER reference implementation

This isolated implementation records the `createHomegrown` behavior in
[RadioFingerprinting](https://github.com/SmartHomePrivacyProject/RadioFingerprinting),
commit `23d1fd4aef2f1dd254c5596680ae1da66f416916`. It does not replace the
previous Phase2 model or make its checkpoints compatible.

The author uses channels-first convolutions followed by Keras 2.2.4's default
channels-last global average pooling. For an input length of 288 the dense
layer therefore receives 288 values; the earlier project port instead reduces
the time axis and feeds it 50. The author model factory and this implementation
were compared on CPU with shared weights. That forward comparison used Keras 3
with its PyTorch backend; it is not a Keras 2 training experiment.

`repository` represents the behavior inferred from the repository's compile
before freeze order and Keras 2.2.4's compile-time collected trainable weights:
the three dense layers are reinitialized and the collected convolution weights
remain trainable. This has not yet been checked by executing a Keras 2.2.4
training step. `paper_last_layer` is a separate interpretation of the paper's
last-layer fine-tuning description, retaining the original head. Both need
independent data, training and prediction-frozen evaluation before either is
called a reproduced paper result.

The paper's five-device cross-day result uses 288 samples and 1,600 support
examples per class. It cannot be compared directly to Phase2's 256 samples,
K=5 or K=10, or the satellite-channel benchmark.
