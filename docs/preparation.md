# Preparation record

Prepared on 2026-11-02 from the existing research workspace.

Retained: the shared solver; ten benchmark drivers; Gmsh/Abaqus inputs for the five benchmark families; original nodal, reaction and timing results; and the latest available checkpoint per retained run folder.

Excluded: manuscript/PPT builders, plotting/digitization scripts, plot exports, figures, documents, unrelated notebooks, editor state, cache files, legacy duplicate benchmark folders, backup FEM directories, nested ZIPs, and per-step checkpoint snapshots.

Removed from the copied numerical drivers: automatic phase-field image generation, final summary plots, and Matplotlib imports. Numerical classes and helper functions were checked against the originals at AST level. The shared module's executable toy demonstration was omitted. The Gmsh reader's description was narrowed to the actual supported ASCII 2.2 format. The original working files were not edited.

Added: an isolated launcher, a result reader that handles appended records, an environment check, dependency specifications, English/Chinese documentation, data catalogs, validation records, and checksums. Original CSV/PT bytes are unchanged and are traceable through `results/file_manifest.csv`.

The launcher defaults the weak-band calculation to the mesh matching the archived primary result, and records any explicit new numerical overrides. This is an input-selection wrapper, not a change to the finite-element or neural-energy equations.

The authors still need to choose the public license. No license or repository URL was assigned by this preparation.
