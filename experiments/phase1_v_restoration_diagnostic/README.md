# Phase1 source V restoration diagnostic

Fixed 4,096-ID sample from the original Phase1 V role. Uses only a frozen Tri-SIG-TIME SSL reader, frozen DURA and the preselected complex-mask checkpoint. Predicts clean and three simulated practical views without source V labels; only after all prediction arrays are sealed does a separate scorer read V truth and report matched accuracy, repairs, damage and TX/RX/day groups. No model or statistics are fitted or updated, no target IQ is read, and no Phase3 work is involved.

The F checkpoint may have used V during historical model selection. The mask training and source IQ selection did not read V. The 4,096 records are a deterministic sample, not the full Phase1 test; simulated satellite is not validated real satellite data.
