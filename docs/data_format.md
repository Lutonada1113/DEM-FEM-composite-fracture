# Numerical data formats

All numerical quantities use the original consistent mm–N–MPa system. Mesh coordinates and displacements are in mm, force is in N, elastic moduli/stress are in MPa, fracture energy is in N/mm, and times are in seconds.

## Nodal field files

`results_*.csv` has the columns:

| Column | Meaning |
| --- | --- |
| `step` | Saved loading increment ID |
| `node` | Original zero-based node ID |
| `x`, `y` | Original nodal coordinates |
| `u_x`, `u_y` | Displacement components |
| `d` | Scalar damage, bounded between 0 and 1 |

Use the node IDs and coordinates carried by each result file. Supplied matching meshes are listed in `field_catalog.json`. Four original CSVs contain repeated `(step,node)` records from appended/resumed runs. `load_results.load_step` retains the last record for that key; it does not change the CSV. Connectivity is not fabricated when an original matching mesh is unavailable.

## Force and timing files

Force files contain `step,U,force`, with `elastic_energy` additionally present in several families. `U` is the imposed displacement. Timing files contain the original step/time fields. Headers in the archived files remain unchanged.

`circlehole1/1.csv` is an original summary table used for the angle-dependent FEM timing comparison. Its stored strength column is not the strength calculated from the primary `fem_alpha*/force_*.csv` files; use the force histories for those strengths. No replacement strength values or missing timing rows were invented during curation.

The historical plotting workflow counted DEM-FEM open-hole times up to the first post-peak increment at which force was nonpositive or at most 5% of the peak. Raw timing histories can include later increments. The archive retains the original histories so the cutoff is explicit in subsequent analysis.

## Checkpoints

The 32 `.pt` files are the latest available saved checkpoints in the retained result folders, rather than one snapshot per loading increment. They contain network state and, depending on the case, damage, history fields, force history, and step times. A checkpoint can precede the last CSV step. Model parameters, loading history, and mesh compatibility must be checked before resuming. New runs do not load these archived checkpoints automatically.

## Catalogs and checksums

- `file_manifest.csv`: byte-preserved original numerical files, original relative paths, sizes, and SHA-256 digests.
- `field_catalog.json`: all saved field increments, row counts, unique node counts, repeated-record counts, damage ranges, and matching mesh candidates.
- `mesh_catalog.json`: supplied Gmsh meshes, node/element counts, and coordinate samples.
- `checkpoint_catalog.json`: available checkpoint steps, state dimensions, and network parameter shapes.
- Root `SHA256SUMS`: release-file checksums, excluding itself.
