# Reproducibility and archived-run provenance

This preparation preserves the numerical model and original data. It does not infer missing historical settings or label every current command as an exact regeneration of an archived run.

## Benchmark families

| Folder | Archived coverage |
| --- | --- |
| `verification` | Two fiber-orientation layouts, A1/A3 fields, A1/A2 equivalence in 2D, and available FEM references. Both A1 CSV versions are retained under their original names. |
| `SENT1` | Fiber angles 30/45/60/90 degrees; beta = 1/10/100 plus baseline 300; an additional 9,890-node DEM30 result. |
| `circlehole1` | Angles 0/15/30/45/60 degrees; length scales 0.2/0.3/0.416/0.8/1.2 mm; mesh parameters 0.1/0.15/0.2/0.3/0.5/0.8/1.2/2 mm; loading-resolution folders; available FEM results with 0.9Gc. |
| `hole` | Equal-area circular, square and triangular openings, with DEM-FEM and FEM fields and reactions. |
| `interface` | Weak-band comparison and additional strong1 archived results. Strong2 remains a supported calculation option but no strong2 results were supplied. |

## Specific differences

1. **SENT30 mesh versions.** `results/SENT1/SENT30/results_SENT30_A1.csv` and its checkpoint have 8,627 nodes. The currently supplied `examples/SENT1/SENT30/SENT30.msh` has 9,890 nodes. The original 8,627-node connectivity was not present in the working folder. A compatible newer result is included under `results/SENT1/SENT30/newmesh_9890/`. The old result retains its own nodal coordinates. Do not pair its node indices with the 9,890-node mesh.

2. **Weak-band mesh.** The primary weak-band fields have 14,365 nodes and 28,498 elements and match `interface_crack.msh`. The legacy `interface_new.msh` is also retained as an input, but the release launcher explicitly selects `interface_crack.msh` for this family. Calling the copied solver directly without the launcher uses that file's original defaults.

3. **Fracture-energy options.** Historical versions used different defaults for effective-Gc correction. The original DEM-FEM open-hole driver enables its effective-Gc option unless `--no-effective-gc` is supplied; its FEM driver enables correction only with `--use-effective-gc`. SENT correction is opt-in; verification has its own `lm` formulation. Use explicit options when comparing methods. The example full open-hole command selects the nominal Gc for DEM-FEM.

4. **Loading schedules and sensitivity runs.** Filenames such as `alpha30-step35` identify an archived sensitivity case; they do not mean that passing only `--steps 35` to the current driver recreates that loading interval. The current `--steps` is an execution limit. The historical `U_MAX`, `N_STEPS`, refinement, stopping parameters, and network training settings were not all retained in each checkpoint. New constant overrides are recorded by the launcher rather than inferred from folder names.

5. **Anisotropy meshes.** SENT45 beta = 1/10/100 results use `SENT45beta.msh`; the baseline beta = 300 uses the ordinary SENT45 mesh. The launcher supports an explicit mesh replacement for a new study. The source material/phase-field functions are unchanged.

6. **Duplicates and checkpoints.** Four CSVs contain repeated `(step,node)` records. The supplied reader resolves them using the original last-write convention. Latest available checkpoints are retained, but not every family has a checkpoint, and a checkpoint may precede the last field increment.

7. **Device and timing.** The provided Python FEM drivers and phase-field sparse solvers use NumPy/SciPy on CPU. CUDA selection affects the neural displacement calculation. Archived timing CSVs are preserved measurements; the original per-run software/hardware and complete command records are not available in those CSVs. The CPU startup checks performed during preparation do not validate archived GPU timing comparisons.

## What was checked

All supplied meshes were parsed. All 81 field CSVs were scanned for schema, saved-step/node counts, and finite bounded damage. After resolving repeated node records, every saved step in each file has the same node count. A matching supplied mesh was identified for 80 field files; the old 8,627-node SENT30 field is the documented exception.

Both solver types were launched after relocation to a separate folder, using the coarse open-hole mesh. The short DEM-FEM run verifies startup, training and data output, not converged accuracy. The one-increment FEM solve verifies sparse assembly/solution and data output. Full fracture simulations were not repeated for this preparation. See `validation.json` for the exact scope and environment.
