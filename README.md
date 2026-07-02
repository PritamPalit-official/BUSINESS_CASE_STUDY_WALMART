<div align="center">

# 🏬 Walmart Business Case Study

### Data-Driven Customer Profiling, Spending Pattern Evaluation & Actionable Marketing Recommendations

*Using EDA, CLT-based Statistical Validation & Conditional Probability Insights*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

*A comprehensive customer spending analysis leveraging transactional data to uncover purchasing behavior patterns and derive data-driven business strategies for Walmart.*

</div>

---

## 📑 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [💼 Business Objective](#-business-objective)
- [📊 Dataset Description](#-dataset-description)
- [🔬 Analysis Workflow](#-analysis-workflow)
- [🛠️ Tools & Technologies](#️-tools--technologies)
- [📁 Project Structure](#-project-structure)
- [💡 Key Insights](#-key-insights)
- [📈 Business Recommendations](#-business-recommendations)
- [🚀 Getting Started](#-getting-started)
- [👤 Author](#-author)
- [📄 License](#-license)

---

## 🎯 Project Overview

This project presents a **comprehensive customer spending analysis** for **Walmart** using transactional data. The objective is to understand purchasing behavior across different customer segments and derive **data-driven recommendations** to support business decision-making.

> The study employs exploratory data analysis (EDA), Central Limit Theorem (CLT)-based statistical validation, and conditional probability techniques to produce portfolio-quality insights suitable for real-world business strategy discussions.

---

## 💼 Business Objective

To analyze customer purchase patterns based on:

| Dimension | Focus Area |
|:--|:--|
| 👫 **Gender** | Male vs. Female spending comparison |
| 🎂 **Age Group** | Age-based segmentation & spending trends |
| 💍 **Marital Status** | Impact of marital status on purchases |
| 🏙️ **City Category** | City-level demand & purchase distribution |

The study evaluates whether **statistically significant differences** exist in spending behavior and translates analytical findings into **actionable business strategies**.

---

## 📊 Dataset Description

The dataset consists of **5,370 customer transactions** with **10 attributes**, enabling detailed segmentation analysis across both categorical and numerical variables.

| # | Feature | Description | Type |
|:-:|:--|:--|:-:|
| 1 | `User_ID` | Unique customer identifier | Numerical |
| 2 | `Product_ID` | Unique product identifier | Categorical |
| 3 | `Gender` | Customer gender (M/F) | Categorical |
| 4 | `Age` | Age group of the customer | Categorical |
| 5 | `Occupation` | Occupation code | Numerical |
| 6 | `City_Category` | City tier (A, B, C) | Categorical |
| 7 | `Stay_In_Current_City_Years` | Duration of stay in current city | Categorical |
| 8 | `Marital_Status` | Marital status (0/1) | Categorical |
| 9 | `Product_Category` | Product category code | Numerical |
| 10 | `Purchase` | Purchase amount (in USD) | Numerical |

---

## 🔬 Analysis Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📥 Data Understanding                                          │
│   └── Dataset shape, data types & statistical summaries         │
│                         ▼                                       │
│  🧹 Data Cleaning                                               │
│   └── Missing value checks, categorical conversion              │
│       & ordered age groups                                      │
│                         ▼                                       │
│  📊 Exploratory Data Analysis                                   │
│   └── Univariate & bivariate visualizations                     │
│                         ▼                                       │
│  📐 Statistical Analysis                                        │
│   └── Confidence intervals & comparison of mean spending        │
│                         ▼                                       │
│  🔍 Segmentation Analysis                                       │
│   └── Gender, age, marital status & city-based comparisons      │
│                         ▼                                       │
│  💡 Insight Generation                                          │
│   └── Business interpretation using CLT principles              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tools & Technologies

| Category | Tools |
|:--|:--|
| **Language** | Python |
| **Data Manipulation** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Statistical Analysis** | SciPy |
| **Environment** | Jupyter Notebook |

---

## 📁 Project Structure

```
BUSINESS_CASE_STUDY_WALMART/
│
├── 📁 data/
│   └── 📊 walmart_data.csv                                            # Transaction dataset
├── 📁 images/                                                         # Exported EDA visualizations
├── 📄 app.py                                                          # Interactive Streamlit dashboard
├── 📄 requirements.txt                                                # Dependency definitions
├── 📓 Walmart_CLT_Analysis.ipynb                                      # Central Limit Theorem Jupyter notebook
├── 🐍 Walmart_CLT_Analysis.py                                         # Python counterpart of the analysis notebook
├── 📊 Defining_Problem_Statement_and_Analyzing_basic_metrics.pdf      # Detailed analysis report
└── 📜 LICENSE                                                         # MIT License
```

---

## 💡 Key Insights

| # | Insight | Statistical Significance |
|:-:|:--|:-:|
| 1 | 🔵 **Male customers** spend more per transaction on average than female customers | ✅ Confirmed (non-overlapping CIs) |
| 2 | 🎯 Customers aged **26–55** represent the **highest spending group** | ✅ Confirmed |
| 3 | 🏙️ **City category** significantly influences overall purchase distribution | ✅ Confirmed |
| 4 | 💍 **Marital status** does not show a significant impact on spending behavior | ❌ Not Significant |

> **Statistical Note:** Confidence intervals were constructed using Central Limit Theorem (CLT) principles to validate the significance of observed spending differences across customer segments.

---

## 📈 Business Recommendations

| # | Recommendation | Target Segment | Expected Impact |
|:-:|:--|:--|:-:|
| 1 | 🎯 Design **targeted marketing campaigns** for high-spending male customer segments | Male, 26–55 | 📈 High |
| 2 | 📢 Focus **promotional strategies** on customers aged 26–55 | Age 26–55 | 📈 High |
| 3 | 📦 **Optimize inventory allocation** based on city-level demand patterns | City Categories A/B/C | 📈 Medium |
| 4 | 🏅 Introduce **loyalty programs** for long-term city residents | Loyal Customers | 📈 Medium |
| 5 | ⚠️ **Avoid over-segmentation** based on marital status due to minimal impact | All Segments | 💰 Cost Saving |

---

## 🚀 Getting Started

### 🖥️ Running the Interactive Dashboard (Streamlit)

To view the live interactive web dashboard:

1. **Clone the repository and navigate inside**:
   ```bash
   git clone https://github.com/PritamPalit-official/BUSINESS_CASE_STUDY_WALMART.git
   cd BUSINESS_CASE_STUDY_WALMART
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

This will automatically launch the dashboard in your default browser at `http://localhost:8501`.

### 📓 Running the Notebook Analysis

If you'd like to explore the step-by-step Jupyter Notebook analysis:

1. **Install Jupyter and base requirements**:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy jupyter
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```
   Open `Walmart_CLT_Analysis.ipynb` to see the complete exploration workflow.

---

## 🛠️ Development & Testing

To maintain production-ready code quality, this repository includes dev dependencies, unit testing configurations, and automated CI pipelines:

### 📦 Setup Developer Dependencies
Install the required development and testing packages:
```bash
pip install -r requirements-dev.txt
```

### 🧪 Run Unit Tests Locally
Run the test suite using Python's built-in `unittest` runner:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### ⚙️ Continuous Integration (CI)
A GitHub Actions workflow is configured in `.github/workflows/ci.yml`. On every `push` and `pull_request` to the repository, it automatically:
1. Provisions an Ubuntu runner with Python 3.10.
2. Installs dependencies from both `requirements.txt` and `requirements-dev.txt`.
3. Runs the test suite to verify code integrity and prevent regressions.

---

## 👤 Author

<div align="center">

**Pritam Palit**

*Electronics & Communication Engineering Graduate*

*Focus Areas: Data Analytics | Statistics | Business Intelligence*

[![GitHub](https://img.shields.io/badge/GitHub-PritamPalit--official-181717?style=for-the-badge&logo=github)](https://github.com/PritamPalit-official)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Pritam_Palit-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/pritam-palit-77b2071b4/)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ *If you found this project insightful, consider giving it a star!* ⭐

</div>
