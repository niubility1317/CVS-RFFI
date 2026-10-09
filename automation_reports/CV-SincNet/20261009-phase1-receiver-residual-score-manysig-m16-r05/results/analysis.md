# Receiver residual: fixed 2×2 comparison

All 16 scratch E200 rows were frozen before query access. Truth opened only after all 16×7 predictions.
Exposed benchmark; six complete practical residual views share physical IDs. Phase2/K/new TX/H: N/A.
No target feedback to tuning, ranking, model selection or selective rerun. All negative results retained.

|Arm|View|Accuracy mean ± SD|Worst RX mean|
|---|---|---:|---:|
|baseline|clean|79.829 ± 1.191%|70.257%|
|displacement|clean|80.532 ± 1.209%|71.219%|
|contribution|clean|79.867 ± 0.981%|71.089%|
|combined|clean|80.404 ± 1.099%|71.208%|
|baseline|practical_high|75.506 ± 2.382%|62.649%|
|displacement|practical_high|75.790 ± 2.205%|63.026%|
|contribution|practical_high|75.592 ± 1.778%|63.279%|
|combined|practical_high|75.844 ± 1.889%|63.803%|
|baseline|practical_mid|73.055 ± 2.041%|61.553%|
|displacement|practical_mid|73.316 ± 1.793%|61.928%|
|contribution|practical_mid|73.110 ± 1.543%|61.934%|
|combined|practical_mid|73.312 ± 1.626%|62.351%|
|baseline|practical_low_suburban|70.767 ± 1.730%|60.377%|
|displacement|practical_low_suburban|70.997 ± 1.473%|60.471%|
|contribution|practical_low_suburban|70.840 ± 1.261%|60.624%|
|combined|practical_low_suburban|71.010 ± 1.311%|61.061%|
|baseline|practical_high_urban|71.968 ± 2.130%|60.227%|
|displacement|practical_high_urban|72.244 ± 1.937%|60.510%|
|contribution|practical_high_urban|72.060 ± 1.619%|60.679%|
|combined|practical_high_urban|72.320 ± 1.706%|61.134%|
|baseline|practical_mid_urban|58.313 ± 1.204%|50.547%|
|displacement|practical_mid_urban|58.545 ± 1.058%|50.840%|
|contribution|practical_mid_urban|58.308 ± 0.896%|50.733%|
|combined|practical_mid_urban|58.512 ± 0.972%|51.069%|
|baseline|practical_low_urban|48.147 ± 0.612%|42.398%|
|displacement|practical_low_urban|48.329 ± 0.518%|42.582%|
|contribution|practical_low_urban|48.076 ± 0.516%|42.450%|
|combined|practical_low_urban|48.197 ± 0.484%|42.711%|
