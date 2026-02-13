# Structural Uncertainty and Yield Equifinality in Crop Models

This repository contains the data and analysis scripts supporting the manuscript:

**Meng, Z., Zhao, H. (2026)**  
*Diagnosing Structural Uncertainty and Yield Equifinality in Process-Based Crop Models: Divergence between APSIM NG and WOFOST for Soybean*  
Submitted to: *Ecological Modelling*

---

## Overview

This study proposes an integrated diagnostic framework to compare structural behaviour between two process-based crop models:

- APSIM Next Generation
- WOFOST (PCSE implementation)

The workflow combines:

1. Harmonised management experiments (44 scenarios)
2. Variance-based global sensitivity analysis (Sobol)
3. SHAP-based meta-model attribution

The results demonstrate **yield equifinality**, where models produce similar yields through different internal process trajectories.

---

## Repository Structure
data/        Input and processed datasets used in the analysis
scripts/     Python scripts used to generate all manuscript figures
outputs/     Generated figures (optional)
---

## Main Figures and Corresponding Scripts

| Figure | Description | Script |
|--------|-------------|--------|
| Fig.1 | Workflow diagram | `Fig7_mechanistic_divergence_layout.py` |
| Fig.2 | Distribution comparisons | `Fig2_year_summary_scatter_boxplot.py` |
| Fig.3 | Seasonal LAI dynamics | `scripts/03_lai_dynamics_fig3.py` |
| Fig.4 | External factor sensitivity | `Fig4_wofost_external_factor_GSA_perm_RF.py` |
| Fig.5 | Structural sensitivity comparison | `Fig5_meta_SHAP_bar_APSIM_WOFOST.py` |
| Fig.6 | SHAP-based attribution | `scripts/06_meta_shap_fig6.py` |

Supplementary figures are generated from scripts in:
scripts/supp/
---

## Reproducing the Figures

All figures can be reproduced using the provided scripts and data.

Example:

```bash
python scripts/03_lai_dynamics_fig3.py
python scripts/06_meta_shap_fig6.py

Requirements

Main Python dependencies:
	•	Python 3.8+
	•	numpy
	•	pandas
	•	matplotlib
	•	scikit-learn
	•	shap

You may install them with:

pip install numpy pandas matplotlib scikit-learn shap

License

This repository is released for academic and research use.

Contact

Zehao Meng
Inner Mongolia University for Nationalities
ORCID: 0009-0009-9669-9357

For questions regarding the data or code, please open an issue in this repository.
