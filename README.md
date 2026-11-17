# DREAMS CDM Milky Way suite — analysis code

Analysis code for **"The DREAMS Project: A New Suite of 1,024 Simulations to Contextualize the
Milky Way and Assess Physics Uncertainties"** (Rose et al.).

The code covers the full analysis chain of the paper:

1. **Galaxy properties.** Present-day properties of the 1,024 central MW-mass galaxies and their
   disk / bulge / stellar-halo decomposition.
2. **GSE analogs.** Identification of Gaia-Sausage-Enceladus (GSE) analog mergers from SubLink
   merger trees.
3. **Emulators.** Conditional normalizing-flow emulators (NeHOD) for the SMHM weighting (§3),
   the GSE / host-property population (§5), low-resolution contamination (App. E) and the
   TNG50/100 comparison (App. D).
4. **Weights.** The SMHM-based parameter weights and every figure and table.

The simulation data are available from the DREAMS project (https://www.dreams-project.org).

## Repository layout

The layout mirrors the original research directory, so the relative paths inside the scripts
work unchanged. Run every script from the directory it lives in. The top-level scripts run
from the repository root.

| Path | Contents | Paper |
|---|---|---|
| `DREAMS.py` | Shared library: I/O, MW selection, ex-situ tracking, disk/bulge/halo decomposition, profiles, plotting helpers, the disk-galaxy sample (`no_disk`, `bad_tree`, `get_disk_sample()`) | §4, App. C |
| `halo_mass.py`, `gas_mass.py`, `vcirc.py`, `bulge_mass.py`, `bulge_vdisp.py`, `disk_size.py`, `halo_shape.py`, `contam.py` | MPI property scripts → `violin_data/*.npy` | §4.1, App. C, App. E |
| `disk_size_new.py`, `compare_methods.py` | Zana et al. (2022) decomposition, used for the §4.1 robustness check | §4.1 |
| `count_disk_particles.py` | Disk-star counts per galaxy, used to define the disk-galaxy sample | §4.1 |
| `calc_gse.py`, `calcGSEprob.ipynb` | GSE-analog quantities from merger trees; builds the GSE-emulator training CSV | §5.1 |
| `violin.py`, `make_table2.py`, `violin_gse.py`, `GSEprob.ipynb` | Figures and Table 2 | Figs 3–5, A1–A4, A7–A8, B1–B4, Table 2 |
| `nehod_weights/` | Weights emulator (5 targets incl. M\*, M_TOT), 30³ parameter grid, SMHM weights | §3, Figs 1–2 |
| `nehod/` | GSE / host-property emulator (27 targets conditioned on M_TOT + 5 parameters) | §5, App. A, App. B |
| `nehod_contam/` | Contamination emulator (+ contamination fraction as a condition); TNG-conditioned sampling | App. E, App. D |
| `TNGdata/analysis/` | TNG50 / TNG100 MW-analog catalogs and the comparison figure | App. D, Fig D1 |
| `obsdata/`, `nehod_weights/obsdata/` | Digitised observational relations: UM-SAGA SMHM (`smhm_{mean,high,low}.csv`), Yang+12, Birrer+14, M–σ, V_circ | §3, §4 |

Expected DREAMS layout (CDM MW-zoom suite, SB5 = 5 varied parameters):

```
$DREAMS_DATA/Sims/CDM/MW_zooms/SB5/box_<N>/snap_090.hdf5
$DREAMS_DATA/FOF_Subfind/CDM/MW_zooms/SB5/box_<N>/fof_subhalo_tab_090.hdf5
$DREAMS_DATA/FOF_Subfind/CDM/MW_zooms/SB5/box_<N>/tree_extended.hdf5      # SubLink
$DREAMS_DATA/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt
```

### Figures and tables

| Paper item | Code | Inputs |
|---|---|---|
| Fig 1 | `nehod_weights/calc_weights_arya.py` → `plots/weights_corner.pdf` | `generated_samples_arya/1000_arya_cdm.npy`, `obsdata/smhm_*.csv` |
| Fig 2 | `nehod_weights/calc_weights_arya.py` → `plots/weights_smhm.pdf` | same |
| Fig 3 | `violin.py` → `plots/violin.pdf` | `violin_data/`, `nehod_weights/arya_weights_binned.npy`, parameter file |
| Table 2 | `make_table2.py` → `table2.csv` | same as Fig 3 |
| Fig 4 | `GSEprob.ipynb` cells 21 (RA) and 24 (GSE) | `nehod/generated_samples/100k_CDM_{Om,s8,SN1,SN2,AGN}_{0-9}_gse_prob.npy` |
| Fig 5 | `GSEprob.ipynb` cell 68 → `plots/gse_comparison.pdf` | `nehod/generated_samples/1000k_CDM_SB_gse_prob.npy` |
| Figs A1, A2, B3, B4 | `GSEprob.ipynb` cell 65; the comment at the top selects the variant | training CSV, emulated samples (B4: plus fixed-TNG-physics samples) |
| Fig A3 | `GSEprob.ipynb` cell 15 | training CSV, emulated samples |
| Fig A4 | `GSEprob.ipynb` cells 15, 17 → `plots/1D_distributions/` | same |
| Fig A7 | `GSEprob.ipynb` cell 72 → `plots/gse_cdf.pdf` | same |
| Fig A8 | `GSEprob.ipynb` cell 74 → `plots/interpolation_vs_extrapolation.pdf` | same |
| Fig B1 | `violin_gse.py` → `plots/violin_GSE.pdf` | `full_data.npy`, `gse_data.npy` (+ weights) from `GSEprob.ipynb` cells 33/38 |
| Fig B2 | `GSEprob.ipynb` cell 77 → `plots/1D_GSE_Comp.pdf` | emulated samples |
| Fig D1 | `TNGdata/analysis/compare_data.py` | `generated_data/flow_*_TNG.npz` and TNG catalogs |
| Fig E2 | `nehod_contam/contam_comparison.py` | `generated_samples/100k_CDM_cont.npy` |
| Fig E3 | `nehod_contam/gse_comparison.py` → `plots/gse_sfr.pdf` | `generated_samples/100k_CDM_SB.npy` (contamination = 0) |

`GSEprob.ipynb` is a research notebook. Several cells build intermediate arrays
(`emulated_data_gse`, `emulated_data_tng`, …) by switching between commented-out lines, so read
each cell before running it.

- **Data and models:** simulation data, derived data products, trained models and generated
  samples.


## Citation

If you use this code, please cite the paper above and the DREAMS (Rose et al. 2025) and NeHOD
(Nguyen et al. 2024) papers.
