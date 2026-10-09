# Receiver residual: fixed 2×2 comparison

All 16 scratch E200 rows were frozen before query access. Truth opened only after all 16×7 predictions.
Exposed benchmark; six complete practical residual views share physical IDs. Phase2/K/new TX/H: N/A.
No target feedback to tuning, ranking, model selection or selective rerun. All negative results retained.

|Arm|View|Accuracy mean ± SD|Worst RX mean|
|---|---|---:|---:|
|baseline|clean|79.755 ± 1.187%|70.317%|
|displacement|clean|80.181 ± 1.189%|70.581%|
|contribution|clean|80.049 ± 1.280%|71.119%|
|combined|clean|80.322 ± 1.007%|70.936%|
|baseline|practical_high|75.759 ± 1.959%|63.067%|
|displacement|practical_high|75.873 ± 2.234%|62.814%|
|contribution|practical_high|75.880 ± 1.893%|63.319%|
|combined|practical_high|76.019 ± 2.181%|63.435%|
|baseline|practical_mid|73.300 ± 1.744%|61.827%|
|displacement|practical_mid|73.398 ± 1.948%|61.553%|
|contribution|practical_mid|73.394 ± 1.693%|61.971%|
|combined|practical_mid|73.533 ± 1.897%|62.111%|
|baseline|practical_low_suburban|71.024 ± 1.478%|60.394%|
|displacement|practical_low_suburban|71.126 ± 1.730%|60.344%|
|contribution|practical_low_suburban|71.092 ± 1.463%|60.633%|
|combined|practical_low_suburban|71.192 ± 1.650%|60.779%|
|baseline|practical_high_urban|72.200 ± 1.817%|60.548%|
|displacement|practical_high_urban|72.350 ± 2.057%|60.275%|
|contribution|practical_high_urban|72.330 ± 1.722%|60.762%|
|combined|practical_high_urban|72.515 ± 2.001%|60.927%|
|baseline|practical_mid_urban|58.474 ± 1.033%|50.755%|
|displacement|practical_mid_urban|58.685 ± 1.245%|50.698%|
|contribution|practical_mid_urban|58.506 ± 1.019%|50.785%|
|combined|practical_mid_urban|58.741 ± 1.186%|51.022%|
|baseline|practical_low_urban|48.261 ± 0.549%|42.465%|
|displacement|practical_low_urban|48.307 ± 0.707%|42.491%|
|contribution|practical_low_urban|48.245 ± 0.572%|42.592%|
|combined|practical_low_urban|48.305 ± 0.646%|42.548%|
