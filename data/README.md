# Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24539834.svg)](https://doi.org/10.5281/zenodo.24539834)

| Folder | In git? | Contents |
|---|---|---|
| `GGL_Catalogs/` | yes | Mock galaxy-galaxy lens catalogs (LSST Y10, 4MOST) |
| `SLSimLensesNoCuts/`, `SLSimLensesNoCutsCatalogs/` | yes | SLSim outputs without selection cuts |
| `skypy_galaxy_catalogs/` | yes | SkyPy galaxy catalogs |
| `posteriors/` | **no, on Zenodo** | emcee HDF5 chains for every forecast and experiment in the paper |
| `samples/` | **no, on Zenodo** | Pairing results and forecast-scenario pickles (mock data) |

## Getting `posteriors/` and `samples/`

These files total about 9 GB, and most are over GitHub's 100 MB limit, so they are archived on Zenodo:

**https://doi.org/10.5281/zenodo.24539834** (Sharma, P., 2026, CC-BY-4.0)

Download every file from the record. Then, from the repository root, run:

```bash
for z in /path/to/download/*.zip; do unzip -n "$z"; done
sha256sum -c /path/to/download/SHA256SUMS
```

This restores:

```
data/posteriors/forecast_w0waCDM_fixed_scatter/   # main forecasts      (notebooks 04a, 05a)
data/posteriors/forecast_w0waCDM_free_scatter/    # inferred scatter    (notebook 04b)
data/posteriors/expt_photoz_precision/            # photo-z precision   (notebook 05b)
data/posteriors/expt_photoz_bias/                 # photo-z accuracy    (notebook 05c)
data/samples/                                     # pickles used by the notebooks
```

The `README.md` on the Zenodo record describes the chain format, including the parameter column order and how to load the chains with emcee.
