# Standard CE clean benchmark figures

The plotting entry point is `tools/figures/plot_standard_ce_clean.py`. It reads existing frozen scores and CSV summaries; it does not train, predict, reopen truth, select checkpoints or rerun experiments.

The source run is `20261010-phase1-standard-ce-baselines-manysig-m56-r01`: 14 architectures, four model seeds (2026092701–2026092704), fixed E200 from scratch with identity cross-entropy only, and the same 168,000 clean target packets per model. This historical comparison is separate from the subsequent E20 three-seed exploration. No satellite results are plotted.

## Reproduction

Use a Python environment with NumPy, matplotlib, Pillow and pypdf. Arial is preferred; the script permits DejaVu Sans or Liberation Sans as font fallbacks. Provide the results directory containing `verified_results.json` and the six result CSV files. The output directory must be new, protecting earlier figure packages.

```text
python tools/figures/plot_standard_ce_clean.py --results <run-results-directory> --output <new-local-figure-directory>
```

Generated PDFs, SVGs, PNGs, preview images, copied source data and archives remain in local artifact storage. They are not committed to Git. The plotting code and this guide are the reproducible Git delivery.

## Figure map

| Figure | Evidence shown | Suggested use |
|---|---|---|
| 1 | All architectures: accuracy and macro-F1, individual seeds, mean and SD | Main clean comparison |
| 2 | Mean accuracy across all seven held-out receiver IDs | Receiver variability |
| 3 | Parameter count versus accuracy, log parameter axis | Capacity/performance comparison |
| 4 | Same-seed 1D minus 2D and complex minus real differences | Architecture comparison, with IQ adaptation caveat |
| 5 | Original versus residual-fusion CVS by receiver and transmitter | Locate gains and negative differences |
| 6 | Mean row-normalized confusion for original CVS, residual-fusion CVS and ResNet-50-1D | Identity-level errors |

Exports use a 183 mm canvas width, editable SVG text, embedded TrueType PDF fonts and 600 dpi PNGs. A six-page PDF and ZIP bundle are also produced. English captions, Chinese usage notes, source CSVs and a manifest accompany the figures. A compact `source_data/verified_results.json` retains only score records, class names and model-seed identifiers, excluding private runtime metadata. This makes the package reproducible with `--results source_data` without remote access. Final journal sizing and font requirements must be checked when a venue is selected.

## Statistics and claim boundary

- Replicates are model seeds, not independent dataset draws. Four seeds share one split and identical physical test packets. Error bars are sample SD (`ddof=1`), not confidence intervals or standard errors.
- All four seeds are retained, including the collapsed Transformer seed 2026092703. No significance stars, hypothesis tests or p-values are supplied.
- Architecture ordering follows mean accuracy for display only. It does not change the preregistered model matrix or select a checkpoint.
- The 2D networks reshape the same IQ packet into 16×16 arrays; dimensionality differences therefore include representation changes. Complex networks and CVS heads also have disclosed adaptations. A common fixed optimizer budget is not an architecture-specific best-tuned comparison.
- Receiver and transmitter negative differences are retained. Heatmap colour limits remain 0–100%. Confusion percentages normalize within each seed before averaging and do not constitute ensemble predictions.
- These are descriptive results from an already exposed proxy benchmark. They do not establish independent blind generalization, satellite-channel robustness or real in-orbit performance.

## Scoped validation

The script reconciles mean and SD against the original summary, retains the registered seeds, checks confusion counts, verifies SVG text elements and extractable PDF text, checks PDF width and PNG DPI, and validates the final archive. Visual inspection is required to catch crowding, clipped labels and ambiguous annotation placement; no experiment rerun is needed to adjust typography.
