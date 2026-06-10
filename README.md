# Learning Food Affordability Representations to Predict Obesity, Diabetes and Healthcare Costs in the US

> **Gilberto Rios & Arul Pandita** | Data Science, Applied Mathematics and Statistics | Johns Hopkins University, Spring 2026

---

## Overview

A data-driven machine learning framework that analyzes food affordability across **~3,100 U.S. counties** to predict obesity rates, diabetes rates, and Medicare costs per capita. The project integrates machine-learned representations, interpretable models, spatial econometrics, and causal inference to move beyond correlation — estimating the **true policy-actionable causal effects** of food affordability on health outcomes.

A key finding is a **$3.50/meal policy tipping point** identified via SHAP analysis, and a policy simulation projecting **over $2.4 billion in potential annual Medicare savings** through targeted interventions across 65 million beneficiaries.

---

## Research Poster

The full capstone poster is available here: [`Capstone_Poster.pdf`](Capstone_Poster.pdf)

---

## Objectives

- Develop data-driven representations of food affordability using machine learning (Autoencoders, PCA)
- Compare learned representations with traditional variables for predictive performance
- Evaluate model interpretability using SHAP analysis
- Estimate causal effects of affordability on health outcomes using Double Machine Learning (DML)
- Assess potential policy impacts through simulation on the Medicare population

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── Capstone_Poster.pdf
│
├── data/
│   ├── raw/                             # Original source data (USDA, CDC, CMS, ACS)
│   │   ├── county_food_costs_2019_2023.csv
│   │   ├── county_grocery_density.csv
│   │   ├── county_median_income.csv
│   │   ├── county_medicare_cost_per_capita_2023.csv
│   │   ├── modified_county_snap_eligibility_gaps.csv
│   │   ├── acs_poverty_2024_clean.csv
│   │   ├── census_population_urbanization_2024_clean.csv
│   │   ├── transportation_proxy_2024_clean.csv
│   │   └── county_minimum_wage_2023_clean.csv
│   │
│   ├── processed/                       # Merged and feature-engineered datasets
│   │   ├── master_merged_8_files.csv
│   │   ├── final_master_with_medicare_2023.csv
│   │   ├── final_master_with_pca.csv
│   │   ├── autoencoder_latent_2d.csv
│   │   ├── autoencoder_latent_3d.csv
│   │   ├── autoencoder_latent_4d.csv
│   │   ├── final_master_with_autoencoder_embeddings.csv
│   │   └── final_spatial_ml_dataset.csv
│   │
│   └── results/                         # Model outputs and causal inference results
│       ├── baselines_results.csv
│       ├── baseline_with_raw_spatial.csv
│       ├── triple_baseline_final_results.csv
│       ├── triple_baseline_NESTED_CV_results.csv
│       ├── ae3d_state_cv_results.csv
│       ├── master_dml_5_treatments.csv
│       ├── all_county_causal_effects.csv
│       └── final_policy_ranking.csv
│
├── notebooks/
│   ├── 01_initial_eda.ipynb
│   ├── 02_individual_eda_analysis.ipynb
│   ├── 03_pca_autoencoder_shap.ipynb
│   ├── 04_ae3d_state_cv_experiments.ipynb
│   ├── 05_dml_causal_inference.ipynb
│   └── 06_dml_heterogeneous_effects.ipynb
│
├── shapefiles/                          # U.S. Census Bureau county boundaries
│   ├── cb_2018_us_county_500k.shp
│   ├── cb_2018_us_county_500k.dbf
│   ├── cb_2018_us_county_500k.shx
│   ├── cb_2018_us_county_500k.prj
│   ├── cb_2018_us_county_500k.cpg
│   ├── cb_2018_us_county_500k_shp_iso.xml
│   └── cb_2018_us_county_500k_shp_ea_iso.xml
│
└── figures/                             # All output visualizations (PNGs)
```

---

## Data Sources

| Dataset | Source | Description |
|---|---|---|
| County Food Costs (2019–2023) | USDA | Cost per meal and food budget shortfall by county |
| Grocery Store Density | USDA | Store counts and per-capita grocery access (GROCPTH20) |
| Median Household Income | ACS | County-level income estimates with confidence intervals |
| Medicare Cost Per Capita (2023) | CMS | Standardized Medicare payment per beneficiary |
| SNAP Eligibility Gaps | ACS | Participation rates and eligibility gaps below poverty line |
| Poverty Rate | ACS | County poverty rates (2024 clean) |
| Population & Urbanization | U.S. Census | Population density and land area |
| Transportation Proxy | ACS | Mean commute time and % households without vehicle |
| Minimum Wage (2023) | State policy data | State minimum wage mapped to county FIPS |

Unit of analysis: **U.S. counties (~3,100)**. All datasets joined on **FIPS county codes**.

---

## Methodology

### 1. Exploratory Data Analysis (`01`, `02`)
Initial and deep-dive EDA across all 9 raw source files. Key findings:
- Food affordability is multi-dimensional: cost, grocery access, transportation, and SNAP participation each contribute independently
- Evidence of non-linear patterns and skewed distributions across affordability variables
- Regional disparities suggest spatial dependence in health outcomes

### 2. Representation Learning: PCA vs. Autoencoders (`03`)

Five core affordability variables were used as input:
- `Cost Per Meal`, `GROCPTH20` (grocery density), `No_SNAP_Below_Poverty_Gap`, `mean_commute_time`, `pct_households_no_vehicle`

**PCA** (linear) was benchmarked against **Autoencoders** (non-linear) across 2D, 3D, and 4D latent spaces using mean squared reconstruction error. Autoencoders outperformed PCA at every dimensionality, confirming that affordability relationships are non-linear.

**Autoencoder Architecture:**
```
Input (5) → Hidden Layer (ReLU) → Bottleneck (2D / 3D / 4D) → Hidden Layer (ReLU) → Output (5)
```

### 3. Spatial Lag Construction (`03`)
County-level autoencoder embeddings were spatially enriched using **Queen contiguity weights** (via `libpysal`) on the U.S. Census shapefile. Spatial lag features (`WX_AE_1`, `WX_AE_2`, `WX_AE_3`) capture the average affordability representation of neighboring counties, encoding regional spillover effects.

### 4. Predictive Modeling (`04`)
Three feature representation experiments were benchmarked using **XGBoost** and **Random Forest** with **State-Blocked Nested Cross-Validation** to ensure geographic generalizability (prevents data leakage through spatial neighbors):

| Experiment | Features |
|---|---|
| Exp 1 | Socioeconomic controls only |
| Exp 2 | Raw food metrics + controls + spatial lags |
| Exp 3 | 3D AE embeddings + controls + spatial lags |

**Best R² results (XGBoost, Exp 3):**
| Outcome | R² |
|---|---|
| Obesity Rate | 0.5912 |
| Diabetes Rate | 0.7301 |
| Medicare Cost Per Capita | 0.2985 |

### 5. SHAP Interpretability (`03`)
SHAP analysis on the best-performing XGBoost model revealed a critical **$3.50/meal policy tipping point** for obesity:
- Below $3.50/meal: lower food costs are associated with *increased* obesity risk
- Above $3.50/meal: the cost variable becomes *protective*, correlating with reduced obesity rates

Transportation access (`pct_households_no_vehicle`) emerged as the single strongest predictor of both obesity and diabetes outcomes.

### 6. Causal Inference via Double Machine Learning (`05`, `06`)

**Method:** `LinearDML` from Microsoft Research's **EconML** library with XGBoost nuisance models and 5-fold cross-fitting.

**Framework:** The residual-on-residual approach orthogonalizes both treatment and outcome against the full confounder set (AE embeddings + spatial lags + socioeconomic controls), isolating the true causal effect from high-dimensional confounding.

**5 treatments × 3 outcomes = 15 causal estimates:**

| Treatment | Outcome | ATE | 95% CI | Significant |
|---|---|---|---|---|
| Lack of Vehicle Access | Obesity Rate (%) | +8.660 | [2.580, 14.740] | ✅ |
| Lack of Vehicle Access | Diabetes Rate (%) | +2.248 | [0.028, 4.469] | ✅ |
| Cost Per Meal | Diabetes Rate (%) | -1.795 | [-2.886, -0.704] | ✅ |
| SNAP Participation Rate | Obesity Rate (%) | +0.119 | [0.074, 0.164] | ✅ |
| SNAP Participation Rate | Diabetes Rate (%) | +0.041 | [0.027, 0.055] | ✅ |

### 7. Policy Simulation

A standardized **10% policy shift** was simulated for each significant lever and mapped onto a base Medicare population of **65 million beneficiaries**:

```
ΔPatients = ATE × ΔPolicy × Population Base
Savings = ΔPatients × Average Annual Care Cost per Patient
```

| Policy Lever | Target Outcome | Projected Annual Medicare Savings |
|---|---|---|
| 10% improvement in vehicle access | Obesity | **$1.046 billion** |
| 10% reduction in cost per meal | Diabetes | **$1.398 billion** |
| 10% increase in SNAP enrollment | Obesity | -$13.3 million (marginal cost increase) |
| **Total net savings** | | **~$2.43 billion/year** |

The SNAP finding highlights a **"Calories vs. Nutrition" trade-off** — simple enrollment expansion without nutritional incentives may not reduce obesity. The study recommends integrating SNAP with produce subsidies to capture multi-billion dollar efficacy.

---

## Key Findings

- **Food affordability is multi-dimensional**: grocery access and transportation matter more than price alone
- **Autoencoders outperform PCA** at capturing non-linear affordability structure
- **Transportation access is the strongest causal driver** of obesity and diabetes outcomes
- **$3.50/meal is a critical policy threshold** — above it, food cost becomes a protective factor
- **$2.4+ billion in annual Medicare savings** are achievable through targeted vehicle access and food subsidy interventions

---

## How to Reproduce

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Learning-Food-Affordability-Representations-to-Predict-Obesity-Diabetes-and-Healthcare-Costs-US.git
cd Learning-Food-Affordability-Representations-to-Predict-Obesity-Diabetes-and-Healthcare-Costs-US
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```
01_initial_eda.ipynb                → EDA on merged dataset
02_individual_eda_analysis.ipynb    → Per-source deep-dive EDA
03_pca_autoencoder_shap.ipynb       → PCA, AE training, spatial lags, SHAP
04_ae3d_state_cv_experiments.ipynb  → Predictive modeling with state-blocked CV
05_dml_causal_inference.ipynb       → DML causal analysis (5 treatments × 3 outcomes)
06_dml_heterogeneous_effects.ipynb  → County-level heterogeneous effects & simulation
```

> **Note:** Notebooks expect data files in the paths defined in each notebook's configuration section. Update `DATA_DIR` and `DATA_PATH` variables if your local structure differs.

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full dependency list. Core libraries:

| Library | Purpose |
|---|---|
| `tensorflow` | Autoencoder training |
| `econml` | Double Machine Learning (DML) |
| `xgboost` | Nuisance models and predictive modeling |
| `shap` | Model interpretability |
| `geopandas` | Shapefile handling |
| `libpysal` | Spatial weights and lag construction |
| `scikit-learn` | Preprocessing, CV, baseline models |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |

---

## Acknowledgements

Thanks to **Anthony Kearsley** and **Dilshad Raihan**, and the Johns Hopkins Department of Applied Mathematics and Statistics.

Data sourced from: **USDA**, **CDC**, **CMS**, **U.S. Census Bureau / ACS**.

---

*Johns Hopkins University — Whiting School of Engineering*
*Data Science, Applied Mathematics and Statistics — Spring 2026*
