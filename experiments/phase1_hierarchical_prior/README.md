# Phase1 hierarchical source Gaussian prior

Fit a two-component waveform submixture within each of six frozen clean-F pseudo classes on the same 8192 source training IDs as the parent prior. Each subcomponent covariance is shrunk toward its class and global source covariance. Compare frozen blind-channel posterior under six versus twelve Gaussian components on fixed source RX8 records, with DURA only as post-fit source reference. No device ground truth, U_s truth, V, or test is read. IQ and high-confidence clean-F pseudo agreement are diagnostics; subcomponents may hallucinate identity. The source sample has been previously explored and is not an independent blind test.

An exploratory four-subcluster-per-class variant fits 24 Gaussian components and compares frozen six-, twelve-, and twenty-four-component priors on identical source records. It uses the same source fit IDs and covariance shrinkage; no TX labels or test are used.

A separate balanced 24-component variant preserves the validated two groups per pseudo class and splits each group at the median along its leading within-group principal component. Minimum 64 source records per subcomponent is retained. It is a distinct algorithm registered after the four-means source preflight found undersized groups.
