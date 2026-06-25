# Strong Lensing Tomography: DSPL and PDSPL Analysis

<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the analysis code, data pipelines, and figure generation scripts necessary to reproduce the results presented in the paper:

**"Strong Lensing Tomography: Double and pseudo multi-source plane strong gravitational lensing to constrain dark energy"** *by Paras Sharma, Simon Birrer, Narayan Khadka, et. al.*

## Overview

This project provides a computational framework to perform hierarchical forecasts on dark energy equation of state parameters. It evaluates traditional Double Source Plane Lenses (DSPLs) alongside a newly proposed methodology: Pseudo Double Source Plane Lenses (PDSPLs). Here we pair deflectors from independent galaxy-galaxy lensing systems to construct PDSPLs, enabling a more extensive sample for cosmological inference.

## Repository Structure

The main analysis workflow is contained within the `notebooks/` directory, supported by the core Python modules in `pdspl_utils/`.

* **`notebooks/`**: The core, sequentially numbered Jupyter notebooks used to run the pipeline:
  * `01_make_skypy_galaxy_catalog.ipynb`: Generates the base background galaxy populations.
  * `02_generate_GGL_catalog.ipynb`: Creates the Galaxy-Galaxy Lens (GGL) mock catalogs for LSST/4MOST.
  * `03_deflector_pairing.ipynb`: Executes the self-similar deflector pairing to construct PDSPLs.
  * `04*_forecast_*.ipynb`: Runs the MCMC hierarchical forecasts under various scatter assumptions.
  * `05*_expt_*.ipynb`: Some hypothetical experiments and their projected constraints.
  * `06*_paper_figures.ipynb`: Generates the final posteriors and figures used in the manuscript.
* **`pdspl_utils/`**: Custom Python package containing the core functionality:
  * `pairing.py`: Algorithm for matching deflectors in galaxy-galaxy lenses.
  * `inference.py`: Likelihood definitions and forecasting pipeline.
  * `data_utils.py` & `plotting.py`: Helper functions for data management and visualization.
* **`data/`**: Stores generated mock catalogs (`GGL_Catalogs/`), SLSim outputs, MCMC posteriors (`posteriors/`), and sample subsets. *(Note: Large files may are be tracked via git).*
* **`figures/`**: Output directory for all generated plots and corner plots found in the paper.
* **`tests/`**: Experimental notebooks for pairing strategies, MCMC convergence, and custom inference pipelines.
* **`paper-latex/`**: LaTeX source code for the submitted manuscript.
* **`roman-lens-finding-proposal/` & `joint-DESC-forecast-stuff/`**: Additional forecasting scenarios, proposals, and exploratory analyses.

## Contacts
Please reach out to paras.sharma@stonybrook.edu for any questions regarding the code or analysis.