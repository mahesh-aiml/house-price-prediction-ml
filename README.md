<div align="center">

# 🏠 House Price Prediction

### Predicting real-estate prices (INR) with Linear, Multiple & Polynomial Regression — plus Gradient Descent implemented from scratch in NumPy

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Dataset](#-dataset)
- [What This Project Covers](#-what-this-project-covers)
- [Visual Results](#-visual-results)
- [Detailed Results](#-detailed-results)
- [Key Learnings & Observations](#-key-learnings--observations)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [Limitations & Future Work](#-limitations--future-work)
- [Author](#-author)

---

## 🎯 Overview

This project is an end-to-end **supervised machine learning** study on a real-estate dataset of **4,200 houses**. The goal is to predict `house_price_inr` from features such as area, bedrooms, bathrooms, location score, age and more.

Rather than just calling `.fit()`, the notebook builds understanding layer by layer:

1. Explore how each feature relates to price.
2. Train **Simple → Multiple → Polynomial** regression models and compare them.
3. Diagnose **underfitting / overfitting** and the **bias–variance** trade-off.
4. Implement **Batch, Stochastic and Mini-Batch Gradient Descent from scratch** (pure NumPy) and compare convergence, speed and accuracy.

---

## 🏆 Key Results

| Model | Features Used | R² | RMSE (₹) |
|---|---|:---:|:---:|
| Simple Linear Regression | `area_sqft` | 0.563 | ≈ 81.8 lakh |
| Multiple Linear Regression | `area_sqft`, `age_years` | 0.568 | ≈ 81.4 lakh |
| Polynomial Regression (deg 2) | `area_sqft` | 0.563 | ≈ 81.8 lakh |
| **Batch GD (from scratch)** | 5 features, scaled | **0.916** | **≈ 35.9 lakh** |
| **Mini-Batch GD (from scratch)** | 5 features, scaled | **0.916** | **≈ 35.9 lakh** |
| SGD (from scratch) | 5 features, scaled | 0.913 | ≈ 36.5 lakh |

> 💡 **Headline insight:** Adding `location_score`, `bedrooms` and `bathrooms` (and scaling the data) lifted R² from **~0.56 to ~0.92**. Price is driven by *more than area alone* — location matters a lot.
> *(Note: the linear/polynomial models and the GD models use different feature sets, so this is a feature-richness comparison, not a like-for-like algorithm comparison. See [Limitations](#-limitations--future-work).)*

---

## 📂 Dataset

**File:** `RealEstate_HousePrice_Dataset_4200.xlsx` — 4,200 rows × 12 columns.

| Column | Description |
|---|---|
| `house_id` | Unique identifier of the house (not used as a predictive feature) |
| `area_sqft` | Built-up area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `location_score` | Desirability score of the location (approx. 1–10) |
| `age_years` | Age of the house in years |
| `distance_city_km` | Distance from the city centre (km) |
| `lot_size_sqft` | Lot / plot size in square feet |
| `has_garage` | 1 if a garage is present, else 0 |
| `has_pool` | 1 if a pool is present, else 0 |
| `renovation_years_ago` | Years since last renovation |
| **`house_price_inr`** | 🎯 **Target** — sale price in Indian Rupees |

**Split:** 80 / 20 train–test (`random_state=42`) → **3,360 training** and **840 testing** samples.

---

## 🧭 What This Project Covers

| # | Topic | What was done |
|---|---|---|
| 1 | **Data Understanding** | Identified independent (features) and dependent (`house_price_inr`) variables |
| 2 | **EDA** | Scatter plots of every feature vs. price |
| 3 | **Train/Test Split** | 80/20 split with a fixed seed for reproducibility |
| 4 | **Simple Linear Regression** | `area_sqft → price`; interpreted slope & intercept |
| 5 | **Assumption Check** | Residual plot to check linearity / constant variance |
| 6 | **Evaluation Metrics** | MSE, MAE, RMSE, R², Adjusted R² |
| 7 | **Multiple Linear Regression** | `area_sqft + age_years`; compared with the simple model |
| 8 | **Polynomial Regression** | Degree 2 vs. linear, visual + numeric comparison |
| 9 | **Over/Underfitting** | Degree sweep 1 → 5 comparing train vs. test error |
| 10 | **Gradient Descent (from scratch)** | Batch, Stochastic & Mini-Batch GD in NumPy |
| 11 | **Convergence & Speed** | Loss curves and training-time comparison |
| 12 | **Bias–Variance Analysis** | Train/Test MSE and error gap across models |
| 13 | **Reporting** | Results exported to `model_evaluation_results.csv` |

---

## 📊 Visual Results

### Feature vs. Price

<table>
<tr>
<td><img src="assets/area_vs_price.png" alt="Area vs Price"/></td>
<td><img src="assets/location_vs_price.png" alt="Location Score vs Price"/></td>
</tr>
<tr>
<td align="center"><b>Area vs. Price</b></td>
<td align="center"><b>Location Score vs. Price</b></td>
</tr>
</table>

### Simple Linear Regression

<table>
<tr>
<td><img src="assets/regression_line.png" alt="Regression line"/></td>
<td><img src="assets/residual_plot.png" alt="Residual plot"/></td>
</tr>
<tr>
<td align="center"><b>Regression line (Area → Price)</b></td>
<td align="center"><b>Residual plot</b></td>
</tr>
</table>

**Fitted equation:**

```
house_price = -1,163,519.18 + 14,788.31 × area_sqft
```

- **Slope ≈ ₹14,788 per sqft** — each extra square foot adds roughly ₹14.8k to the predicted price.
- **Intercept ≈ −₹11.6 lakh** — a mathematical offset (a 0-sqft house isn't meaningful).

### Linear vs. Polynomial

<img src="assets/linear_vs_polynomial.png" alt="Linear vs Polynomial" width="640"/>

### Gradient Descent Convergence

<img src="assets/gd_convergence.png" alt="Gradient descent convergence" width="640"/>

---

## 📈 Detailed Results

### 1️⃣ Simple vs. Multiple Linear Regression

| Metric | Simple (`area`) | Multiple (`area` + `age`) |
|---|---:|---:|
| MSE | 6.699 × 10¹³ | 6.619 × 10¹³ |
| MAE (₹) | 62,94,594 | 62,37,589 |
| RMSE (₹) | 81,84,697 | 81,35,693 |
| R² | 0.5625 | 0.5677 |

### 2️⃣ Polynomial Degree Sweep (feature: `area_sqft`)

| Degree | Train MSE | Test MSE | Gap |
|:---:|---:|---:|---:|
| 1 | 6.568 × 10¹³ | 6.699 × 10¹³ | 1.31 × 10¹² |
| **2** ✅ | **6.566 × 10¹³** | **6.696 × 10¹³** | 1.31 × 10¹² |
| 3 | 6.615 × 10¹³ | 6.734 × 10¹³ | 1.19 × 10¹² |
| 4 | 6.927 × 10¹³ | 7.017 × 10¹³ | 9.02 × 10¹¹ |
| 5 | 7.548 × 10¹³ | 7.621 × 10¹³ | 7.28 × 10¹¹ |

➡️ **Degree 2** gives the lowest test error, but the gain over a straight line is tiny — the area–price relationship is essentially linear.

### 3️⃣ Gradient Descent Variants (5 scaled features)

Features: `area_sqft`, `bedrooms`, `bathrooms`, `location_score`, `age_years`

| Method | Hyper-parameters | Train Time | MAE (₹) | RMSE (₹) | R² |
|---|---|---:|---:|---:|---:|
| **Batch GD** | lr = 0.01, 1000 epochs | **0.048 s** | 26,43,369 | 35,90,027 | **0.9158** |
| **Stochastic GD** | lr = 0.001, 50 epochs | 0.684 s | 26,59,510 | 36,52,018 | 0.9129 |
| **Mini-Batch GD** | lr = 0.01, 100 epochs, batch = 32 | 0.122 s | **26,42,581** | 35,92,618 | 0.9157 |

- **Batch GD** — most stable, smooth convergence, fastest here (vectorised over a small dataset).
- **SGD** — noisiest updates and slowest (a Python loop over every sample).
- **Mini-Batch GD** — best balance of stability and update frequency; matches Batch GD's accuracy.

### 4️⃣ Bias–Variance Diagnostics

| Model | Train MSE | Test MSE | Gap (variance proxy) |
|---|---:|---:|---:|
| Simple Linear | 6.568 × 10¹³ | 6.699 × 10¹³ | 1.31 × 10¹² |
| Multiple Linear | 6.496 × 10¹³ | 6.619 × 10¹³ | 1.23 × 10¹² |
| Polynomial (deg 2) | 6.566 × 10¹³ | 6.696 × 10¹³ | 1.31 × 10¹² |

Train and test errors are close **and both are high** → the models suffer from **high bias (underfitting)**, not high variance. The fix is *more informative features*, not more model complexity.

---

## 🧠 Key Learnings & Observations

1. **Area alone is a weak predictor** — it explains only ~56% of the variance in price.
2. **Location matters** — the `location_score` scatter plot shows a clear upward trend, and including it (with bedrooms/bathrooms) pushed R² above 0.91.
3. **Higher-degree polynomials don't help** — degree 2 is the best, and beyond that test error rises.
4. **Feature scaling is essential for gradient descent** — inputs and target were standardised (`StandardScaler` for X, manual z-score for y), and predictions were converted back to INR.
5. **All three GD flavours converge to nearly the same solution**; they differ in stability, update cost and wall-clock time.
6. **Underfitting was the main issue** — a small train–test gap combined with high error is the signature of high bias.

---

## 🗂 Project Structure

```
House-Price-Prediction/
│
├── 📓 House_Price_Prediction.ipynb            # Main notebook (analysis + models)
├── 📊 model_evaluation_results.csv            # Exported bias–variance results
├── 📁 data/
│   └── RealEstate_HousePrice_Dataset_4200.xlsx   # Dataset (add here)
├── 📁 assets/                                 # Plots used in this README
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset and fix the path

Place `RealEstate_HousePrice_Dataset_4200.xlsx` inside the `data/` folder, then update the loading line in the notebook to a **relative path**:

```python
data = pd.read_excel("data/RealEstate_HousePrice_Dataset_4200.xlsx")
```

### 5. Run the notebook

```bash
jupyter notebook House_Price_Prediction.ipynb
```

---

## 🛠 Tech Stack

| Purpose | Library |
|---|---|
| Data handling | `pandas`, `numpy`, `openpyxl` |
| Visualisation | `matplotlib`, `seaborn` |
| Modelling | `scikit-learn` (`LinearRegression`, `PolynomialFeatures`, `StandardScaler`, metrics) |
| From-scratch algorithms | Pure `NumPy` (Batch / SGD / Mini-Batch GD) |
| Environment | Jupyter Notebook |

---

## ⚠️ Limitations & Future Work

Being upfront about what could be improved:

- **Feature sets differ between experiments.** Simple/Multiple/Polynomial models use 1–2 features while the GD models use 5, so R² is not directly comparable. *Next step:* run every model on the same feature set.
- **Unscaled polynomial features.** The R² drop at degree 5 is likely partly numerical instability from unscaled powers, not just overfitting. *Next step:* use a `Pipeline` with `StandardScaler` + `PolynomialFeatures`.
- **`house_id` is an identifier**, not a real predictor — it should be excluded from any feature matrix.
- **Adjusted R² for the multiple-regression model** should use that model's own R² and feature count when compared side by side.
- **Bias/variance are proxies** (train MSE and train–test gap), not the formal decomposition. Bootstrapping would give a rigorous estimate.
- **Hard-coded local file path** — replace it with a relative path (see [Getting Started](#-getting-started)).

**Ideas to extend the project:**

- [ ] Use all relevant features in a single multiple regression model
- [ ] Add **Ridge / Lasso / ElasticNet** regularisation
- [ ] Try **Random Forest / Gradient Boosting / XGBoost**
- [ ] Add **k-fold cross-validation** and hyper-parameter tuning
- [ ] Add a correlation heatmap and outlier analysis
- [ ] Deploy a small **Streamlit / Flask** app for live predictions

---

## 👤 Author

**[Mahesh Lohar]**

- LinkedIn: [your-profile](www.linkedin.com/in/mahesh2211)

---

<div align="center">

⭐ If you found this project useful, please consider giving it a star!

</div>
