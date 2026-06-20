# %% [markdown]
# # 🏬 Walmart — Confidence Interval & Central Limit Theorem Analysis
#
# **Author:** Pritam Palit
# **Date:** June 2026
#
# ---
#
# ## Business Context
#
# Walmart is one of the largest retail corporations in the world. The management
# wants to analyse customer purchase behaviour (specifically, **purchase amount**)
# against various customer demographics — **Gender, Age, Marital Status** — during
# **Black Friday** sales events.
#
# Understanding whether spending habits truly differ across these segments (and
# by *how much*) is critical for targeted marketing, inventory planning, and
# personalised offers.
#
# We will leverage the **Central Limit Theorem (CLT)** and **Confidence Intervals**
# to draw statistically sound conclusions from sample data.

# %%
# ── Imports & Configuration ──────────────────────────────────────────────────
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'font.size': 11,
    'figure.dpi': 120,
})

IMG_DIR = 'images'
DATA_PATH = os.path.join('data', 'walmart_data.csv')
os.makedirs(IMG_DIR, exist_ok=True)

print("✅ Libraries loaded successfully.")

# %% [markdown]
# ---
# # 1. Problem Statement & Basic Metrics *(10 pts)*
#
# > **Problem:** Walmart wants to understand if spending habits differ between
# > male and female customers during Black Friday. The company assumes roughly
# > **50 million male** and **50 million female** customers. We need to determine
# > whether any observed difference in average spending is statistically
# > significant or merely due to sampling variation.
#
# ### Objectives
# 1. Explore the dataset — shape, types, missing values, summary statistics.
# 2. Perform visual & non-graphical EDA.
# 3. Apply the **Central Limit Theorem** to build confidence intervals.
# 4. Answer whether Male vs Female, Married vs Unmarried, and different Age
#    groups have significantly different spending patterns.

# %%
# ── 1.1  Load the Dataset ───────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
df.head(10)

# %%
# ── 1.2  Shape & Column Types ───────────────────────────────────────────────
print(f"Shape : {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")

# %%
# ── 1.3  Missing Values ─────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df)
print(f"\nTotal missing cells: {df.isnull().sum().sum()}")

# %%
# ── 1.4  Statistical Summary ────────────────────────────────────────────────
df.describe(include='all').T

# %%
# ── 1.5  Data Info ───────────────────────────────────────────────────────────
df.info()

# %% [markdown]
# ---
# # 2. Non-Graphical & Visual Analysis *(10 pts)*

# %%
# ── 2.1  Value Counts & Unique Attributes ────────────────────────────────────
cat_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years',
            'Marital_Status', 'Occupation', 'Product_Category']

for col in cat_cols:
    print(f"\n{'='*50}")
    print(f"  {col}  — {df[col].nunique()} unique values")
    print(f"{'='*50}")
    print(df[col].value_counts())

# %%
# ── 2.2  Distribution of Purchase Amount ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(df['Purchase'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title('Histogram — Purchase Amount')
axes[0].set_xlabel('Purchase Amount ($)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['Purchase'].mean(), color='red', ls='--', lw=2, label=f"Mean = ${df['Purchase'].mean():,.0f}")
axes[0].legend()

sns.kdeplot(df['Purchase'], ax=axes[1], fill=True, color='darkorange', alpha=0.5)
axes[1].set_title('KDE — Purchase Amount')
axes[1].set_xlabel('Purchase Amount ($)')
axes[1].axvline(df['Purchase'].mean(), color='red', ls='--', lw=2, label=f"Mean = ${df['Purchase'].mean():,.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '01_purchase_distribution.png'), bbox_inches='tight')
plt.show()

# %%
# ── 2.3  Count Plots — Gender, Age, City, Marital Status ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

sns.countplot(data=df, x='Gender', palette='Set2', ax=axes[0, 0], edgecolor='black')
axes[0, 0].set_title('Transactions by Gender')
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

order_age = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']
sns.countplot(data=df, x='Age', order=order_age, palette='coolwarm', ax=axes[0, 1], edgecolor='black')
axes[0, 1].set_title('Transactions by Age Group')
axes[0, 1].tick_params(axis='x', rotation=30)
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

sns.countplot(data=df, x='City_Category', palette='pastel', ax=axes[1, 0], edgecolor='black')
axes[1, 0].set_title('Transactions by City Category')
for p in axes[1, 0].patches:
    axes[1, 0].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

sns.countplot(data=df, x='Marital_Status', palette='Set1', ax=axes[1, 1], edgecolor='black')
axes[1, 1].set_title('Transactions by Marital Status')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(['Unmarried (0)', 'Married (1)'])
for p in axes[1, 1].patches:
    axes[1, 1].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Count Plots — Transaction Distributions', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '02_count_plots.png'), bbox_inches='tight')
plt.show()

# %%
# ── 2.4  Box Plots — Purchase by Gender, Age, City ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.boxplot(data=df, x='Gender', y='Purchase', palette='Set2', ax=axes[0])
axes[0].set_title('Purchase by Gender')
axes[0].set_ylabel('Purchase Amount ($)')

sns.boxplot(data=df, x='Age', y='Purchase', order=order_age, palette='coolwarm', ax=axes[1])
axes[1].set_title('Purchase by Age Group')
axes[1].tick_params(axis='x', rotation=30)
axes[1].set_ylabel('Purchase Amount ($)')

sns.boxplot(data=df, x='City_Category', y='Purchase', palette='pastel', ax=axes[2])
axes[2].set_title('Purchase by City Category')
axes[2].set_ylabel('Purchase Amount ($)')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '03_boxplots_purchase.png'), bbox_inches='tight')
plt.show()

# %%
# ── 2.5  Correlation Heatmap ────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            mask=mask, linewidths=0.5, ax=ax, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap — Numeric Features')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '04_correlation_heatmap.png'), bbox_inches='tight')
plt.show()

# %%
# ── 2.6  Pair Plot (Numeric Features) ───────────────────────────────────────
sample_pair = df[numeric_cols + ['Gender']].sample(n=min(2000, len(df)), random_state=42)
g = sns.pairplot(sample_pair, hue='Gender', palette='Set1', diag_kind='kde',
                 plot_kws={'alpha': 0.4, 's': 15})
g.figure.suptitle('Pair Plot — Numeric Features by Gender', y=1.02, fontsize=14)
plt.savefig(os.path.join(IMG_DIR, '05_pairplot.png'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# # 3. Missing Value & Outlier Detection *(10 pts)*

# %%
# ── 3.1  Missing Value Summary ──────────────────────────────────────────────
print("=" * 50)
print("   MISSING VALUE REPORT")
print("=" * 50)
total_cells = df.shape[0] * df.shape[1]
total_missing = df.isnull().sum().sum()
print(f"Total cells        : {total_cells:,}")
print(f"Total missing      : {total_missing:,}")
print(f"Missing percentage : {total_missing / total_cells * 100:.2f}%")
print()
for col in df.columns:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        print(f"  ⚠️  {col}: {n_miss:,} missing ({n_miss/len(df)*100:.2f}%)")
if total_missing == 0:
    print("  ✅ No missing values found in any column!")

# %%
# ── 3.2  Outlier Detection using IQR Method ────────────────────────────────
print("=" * 50)
print("   OUTLIER DETECTION (IQR Method)")
print("=" * 50)

numeric_for_outlier = ['Purchase', 'Occupation', 'Product_Category']

outlier_summary = []
for col in numeric_for_outlier:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary.append({
        'Column': col, 'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'Lower Bound': lower, 'Upper Bound': upper,
        'Outliers': n_outliers, 'Outlier %': round(n_outliers/len(df)*100, 2)
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))

# %%
# ── 3.3  Visualise Outliers ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(numeric_for_outlier), figsize=(6*len(numeric_for_outlier), 5))
if len(numeric_for_outlier) == 1:
    axes = [axes]

for i, col in enumerate(numeric_for_outlier):
    sns.boxplot(y=df[col], ax=axes[i], color='lightcoral', width=0.4)
    axes[i].set_title(f'Outlier Box Plot — {col}')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '06_outlier_boxplots.png'), bbox_inches='tight')
plt.show()

print("\n💡 Note: We will NOT remove outliers as they represent genuine high-value"
      "\n   purchases during Black Friday — removing them would bias our analysis.")

# %% [markdown]
# ---
# # 4. Business Insights from Visual Analysis *(10 pts)*
#
# Based on the EDA above, we can draw several preliminary insights:
#
# ### 👤 Gender
# - **Male customers dominate** the transaction count — roughly **75%** of all
#   purchases are made by males.
# - However, *more transactions ≠ higher per-transaction spending*. We need
#   statistical tests (CI analysis) to confirm any spending difference.
#
# ### 📊 Age Group
# - The **26-35** age group contributes the most transactions, followed by
#   **36-45** and **18-25**.
# - Younger (0-17) and older (55+) segments contribute the fewest transactions.
#
# ### 🏙️ City Category
# - **City B** has the highest transaction count, followed by **City C** and
#   **City A**.
# - Urban tier-2 cities appear to be the primary revenue drivers.
#
# ### 💍 Marital Status
# - **Unmarried customers** make slightly more transactions than married
#   customers.
#
# ### 💰 Purchase Amount
# - The purchase distribution is **right-skewed** with a long tail.
# - Most purchases cluster between **$5,000–$15,000**.
# - There are notable high-value outliers — genuine Black Friday big-ticket
#   purchases.
#
# ### 🔗 Correlations
# - Most numerical features show **weak correlations** with Purchase, indicating
#   that demographics alone don't fully explain purchase behaviour.

# %% [markdown]
# ---
# # 5. Answering the Key Business Questions *(50 pts)*
#
# ---
# ## 5.1 Are women spending more per transaction than men? *(10 pts)*

# %%
# ── 5.1.1  Average Spending by Gender ────────────────────────────────────────
gender_stats = df.groupby('Gender')['Purchase'].agg(['mean', 'median', 'std', 'count'])
gender_stats.columns = ['Mean ($)', 'Median ($)', 'Std Dev ($)', 'Num Transactions']
gender_stats['Mean ($)'] = gender_stats['Mean ($)'].round(2)
gender_stats['Median ($)'] = gender_stats['Median ($)'].round(2)
gender_stats['Std Dev ($)'] = gender_stats['Std Dev ($)'].round(2)
print("=" * 65)
print("   AVERAGE SPENDING BY GENDER")
print("=" * 65)
print(gender_stats)
print()

male_mean = df[df['Gender'] == 'M']['Purchase'].mean()
female_mean = df[df['Gender'] == 'F']['Purchase'].mean()
diff = male_mean - female_mean

print(f"Male   avg spend : ${male_mean:,.2f}")
print(f"Female avg spend : ${female_mean:,.2f}")
print(f"Difference       : ${diff:,.2f} ({'Males spend more' if diff > 0 else 'Females spend more'})")

# %%
# ── 5.1.2  Purchase per Transaction — Male vs Female ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Histogram overlay
for g, color, label in [('M', 'steelblue', 'Male'), ('F', 'salmon', 'Female')]:
    axes[0].hist(df[df['Gender'] == g]['Purchase'], bins=50, alpha=0.6,
                 color=color, label=label, edgecolor='white')
axes[0].axvline(male_mean, color='steelblue', ls='--', lw=2, label=f'Male Mean ${male_mean:,.0f}')
axes[0].axvline(female_mean, color='salmon', ls='--', lw=2, label=f'Female Mean ${female_mean:,.0f}')
axes[0].set_title('Purchase Distribution — Male vs Female')
axes[0].set_xlabel('Purchase Amount ($)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Bar chart of means
bars = axes[1].bar(['Male', 'Female'], [male_mean, female_mean],
                    color=['steelblue', 'salmon'], edgecolor='black', width=0.5)
axes[1].set_title('Average Purchase — Male vs Female')
axes[1].set_ylabel('Average Purchase ($)')
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                 f'${bar.get_height():,.0f}', ha='center', va='bottom', fontweight='bold')
axes[1].set_ylim(0, max(male_mean, female_mean) * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '07_gender_spending_comparison.png'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 📌 Inference — Section 5.1
#
# From the sample data, **males appear to spend more per transaction** than females
# on average. However, this is a *point estimate* from a single sample — we cannot
# conclude that the difference is statistically significant without confidence
# intervals and hypothesis testing.
#
# ➡️ Let's use the **Central Limit Theorem** to quantify the uncertainty.

# %% [markdown]
# ---
# ## 5.2 Confidence Intervals for Male & Female Spending *(10 pts)*
#
# ### Central Limit Theorem (CLT) Recap
#
# If we draw random samples of size $n$ from a population with mean $\mu$ and
# standard deviation $\sigma$, then as $n \to \infty$, the **distribution of
# sample means** approaches a **Normal distribution**:
#
# $$\bar{X} \sim \mathcal{N}\left(\mu,\; \frac{\sigma}{\sqrt{n}}\right)$$
#
# This holds regardless of the shape of the original population distribution,
# provided $n$ is sufficiently large (typically $n \geq 30$).

# %%
# ── 5.2.1  CLT Demonstration — Sampling Distribution of Means ───────────────
np.random.seed(42)

male_purchases = df[df['Gender'] == 'M']['Purchase'].values
female_purchases = df[df['Gender'] == 'F']['Purchase'].values

sample_sizes = [30, 100, 500, 1000]
n_simulations = 1000

fig, axes = plt.subplots(len(sample_sizes), 2, figsize=(16, 5 * len(sample_sizes)))

for i, n in enumerate(sample_sizes):
    male_means = [np.random.choice(male_purchases, size=n, replace=True).mean()
                  for _ in range(n_simulations)]
    female_means = [np.random.choice(female_purchases, size=n, replace=True).mean()
                    for _ in range(n_simulations)]

    axes[i, 0].hist(male_means, bins=40, color='steelblue', alpha=0.75, edgecolor='white')
    axes[i, 0].axvline(np.mean(male_means), color='red', ls='--', lw=2,
                        label=f'Mean of means = ${np.mean(male_means):,.0f}')
    axes[i, 0].set_title(f'Male — Sample Means Distribution  (n = {n})')
    axes[i, 0].set_xlabel('Sample Mean ($)')
    axes[i, 0].legend()

    axes[i, 1].hist(female_means, bins=40, color='salmon', alpha=0.75, edgecolor='white')
    axes[i, 1].axvline(np.mean(female_means), color='red', ls='--', lw=2,
                        label=f'Mean of means = ${np.mean(female_means):,.0f}')
    axes[i, 1].set_title(f'Female — Sample Means Distribution  (n = {n})')
    axes[i, 1].set_xlabel('Sample Mean ($)')
    axes[i, 1].legend()

plt.suptitle('Central Limit Theorem — Distribution of Sample Means at Various Sample Sizes',
             fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '08_clt_sampling_distributions.png'), bbox_inches='tight')
plt.show()

print("✅ As sample size increases, the distribution of sample means becomes")
print("   narrower and more normally distributed — confirming CLT.")

# %%
# ── 5.2.2  Confidence Interval Calculation Function ─────────────────────────
def compute_confidence_interval(data, confidence_levels=[0.90, 0.95, 0.99],
                                 sample_sizes=[30, 100, 500, 1000],
                                 n_bootstrap=1000, seed=42):
    """
    Compute CIs using the CLT for various sample sizes and confidence levels.
    Uses bootstrap sampling to estimate the standard error.
    """
    np.random.seed(seed)
    results = []

    pop_mean = data.mean()
    pop_std = data.std()

    for n in sample_sizes:
        # Bootstrap sample means
        sample_means = [np.random.choice(data, size=n, replace=True).mean()
                        for _ in range(n_bootstrap)]
        se = np.std(sample_means, ddof=1)  # standard error of sample means

        for conf in confidence_levels:
            z = stats.norm.ppf((1 + conf) / 2)
            margin = z * se
            ci_lower = pop_mean - margin
            ci_upper = pop_mean + margin

            results.append({
                'Sample Size': n,
                'Confidence Level': f'{conf*100:.0f}%',
                'Population Mean': round(pop_mean, 2),
                'Std Error': round(se, 2),
                'Z-score': round(z, 4),
                'Margin of Error': round(margin, 2),
                'CI Lower': round(ci_lower, 2),
                'CI Upper': round(ci_upper, 2),
                'CI Width': round(ci_upper - ci_lower, 2),
            })

    return pd.DataFrame(results)


# Compute CIs for Males
print("=" * 80)
print("   CONFIDENCE INTERVALS — MALE CUSTOMERS")
print("=" * 80)
ci_male = compute_confidence_interval(male_purchases)
print(ci_male.to_string(index=False))

# %%
# Compute CIs for Females
print("=" * 80)
print("   CONFIDENCE INTERVALS — FEMALE CUSTOMERS")
print("=" * 80)
ci_female = compute_confidence_interval(female_purchases)
print(ci_female.to_string(index=False))

# %%
# ── 5.2.3  Visualise CIs for Males and Females ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

conf_levels = ['90%', '95%', '99%']
colors_m = ['#1f77b4', '#2ca02c', '#d62728']
colors_f = ['#ff7f0e', '#9467bd', '#8c564b']

for idx, conf in enumerate(conf_levels):
    ax = axes[idx]
    ci_m_sub = ci_male[ci_male['Confidence Level'] == conf]
    ci_f_sub = ci_female[ci_female['Confidence Level'] == conf]

    y_positions_m = np.arange(len(ci_m_sub)) * 2 + 0.3
    y_positions_f = np.arange(len(ci_f_sub)) * 2 - 0.3

    for j, (_, row) in enumerate(ci_m_sub.iterrows()):
        ax.plot([row['CI Lower'], row['CI Upper']], [y_positions_m[j]]*2,
                'o-', color=colors_m[idx], lw=2.5, markersize=6)
        ax.plot(row['Population Mean'], y_positions_m[j], 's',
                color=colors_m[idx], markersize=8)

    for j, (_, row) in enumerate(ci_f_sub.iterrows()):
        ax.plot([row['CI Lower'], row['CI Upper']], [y_positions_f[j]]*2,
                'o-', color=colors_f[idx], lw=2.5, markersize=6)
        ax.plot(row['Population Mean'], y_positions_f[j], 's',
                color=colors_f[idx], markersize=8)

    ax.set_yticks(np.arange(len(ci_m_sub)) * 2)
    ax.set_yticklabels([f'n = {int(s)}' for s in ci_m_sub['Sample Size']])
    ax.set_title(f'{conf} Confidence Interval')
    ax.set_xlabel('Purchase Amount ($)')
    ax.legend(['Male CI', 'Male Mean', 'Female CI', 'Female Mean'],
              loc='upper right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Confidence Intervals — Male (Blue/Green/Red) vs Female (Orange/Purple/Brown)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '09_confidence_intervals_gender.png'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 5.3 Do CIs Overlap? Business Leverage? *(10 pts)*

# %%
# ── 5.3.1  Overlap Analysis — Male vs Female ────────────────────────────────
print("=" * 90)
print("   CI OVERLAP ANALYSIS — MALE vs FEMALE")
print("=" * 90)

overlap_results = []

for conf in ['90%', '95%', '99%']:
    for n in [30, 100, 500, 1000]:
        m_row = ci_male[(ci_male['Confidence Level'] == conf) &
                        (ci_male['Sample Size'] == n)].iloc[0]
        f_row = ci_female[(ci_female['Confidence Level'] == conf) &
                          (ci_female['Sample Size'] == n)].iloc[0]

        overlap = m_row['CI Lower'] <= f_row['CI Upper'] and f_row['CI Lower'] <= m_row['CI Upper']
        overlap_amount = 0
        if overlap:
            overlap_amount = min(m_row['CI Upper'], f_row['CI Upper']) - max(m_row['CI Lower'], f_row['CI Lower'])

        overlap_results.append({
            'Confidence': conf,
            'Sample Size': n,
            'Male CI': f"[{m_row['CI Lower']:,.0f}, {m_row['CI Upper']:,.0f}]",
            'Female CI': f"[{f_row['CI Lower']:,.0f}, {f_row['CI Upper']:,.0f}]",
            'Overlap?': '✅ Yes' if overlap else '❌ No',
            'Overlap Width': round(overlap_amount, 0) if overlap else 0,
        })

overlap_df = pd.DataFrame(overlap_results)
print(overlap_df.to_string(index=False))

# %%
# ── 5.3.2  Visualise CI Overlap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

ci_m_95 = ci_male[ci_male['Confidence Level'] == '95%']
ci_f_95 = ci_female[ci_female['Confidence Level'] == '95%']

y_labels = []
y_pos = 0
for _, (m_row, f_row) in enumerate(zip(ci_m_95.itertuples(), ci_f_95.itertuples())):
    ax.barh(y_pos + 0.2, m_row._8 - m_row._7, left=m_row._7, height=0.35,
            color='steelblue', alpha=0.7, edgecolor='black', label='Male' if y_pos == 0 else '')
    ax.barh(y_pos - 0.2, f_row._8 - f_row._7, left=f_row._7, height=0.35,
            color='salmon', alpha=0.7, edgecolor='black', label='Female' if y_pos == 0 else '')
    ax.plot(m_row._3, y_pos + 0.2, 'D', color='navy', markersize=8)
    ax.plot(f_row._3, y_pos - 0.2, 'D', color='darkred', markersize=8)
    y_labels.append(f'n = {int(m_row._1)}')
    y_pos += 1.2

ax.set_yticks([i * 1.2 for i in range(len(y_labels))])
ax.set_yticklabels(y_labels, fontsize=12)
ax.set_xlabel('Purchase Amount ($)', fontsize=12)
ax.set_title('95% Confidence Intervals — Male vs Female (by Sample Size)', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '10_ci_overlap_gender.png'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 📌 Interpretation — CI Overlap
#
# | Observation | Detail |
# |---|---|
# | **Small samples (n=30)** | CIs are wide and overlap substantially → cannot conclude spending differs. |
# | **Larger samples (n≥500)** | CIs become narrower; overlap may decrease or disappear. |
# | **If CIs overlap** | We cannot conclusively say that males and females spend differently *at that confidence level*. |
# | **If CIs do NOT overlap** | There is strong statistical evidence that spending truly differs. |
#
# ### 🏢 How Walmart Can Leverage This
# - **If no significant difference**: Apply a *uniform marketing strategy* across
#   genders during Black Friday — same promotions, same budget allocation.
# - **If significant difference**: Deploy *gender-targeted campaigns* — allocate
#   more marketing budget toward the higher-spending segment and create
#   personalised offers for the lower-spending segment to boost their spend.

# %% [markdown]
# ---
# ## 5.4 Married vs Unmarried Analysis *(10 pts)*

# %%
# ── 5.4.1  Descriptive Stats — Marital Status ──────────────────────────────
marital_stats = df.groupby('Marital_Status')['Purchase'].agg(['mean', 'median', 'std', 'count'])
marital_stats.index = ['Unmarried (0)', 'Married (1)']
marital_stats.columns = ['Mean ($)', 'Median ($)', 'Std Dev ($)', 'Count']
print("=" * 65)
print("   SPENDING BY MARITAL STATUS")
print("=" * 65)
print(marital_stats.round(2))

unmarried_purchases = df[df['Marital_Status'] == 0]['Purchase'].values
married_purchases = df[df['Marital_Status'] == 1]['Purchase'].values

# %%
# ── 5.4.2  Visualise Marital Status Spending ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ms, color, label in [(0, 'teal', 'Unmarried'), (1, 'coral', 'Married')]:
    axes[0].hist(df[df['Marital_Status'] == ms]['Purchase'], bins=50, alpha=0.6,
                 color=color, label=label, edgecolor='white')
axes[0].set_title('Purchase Distribution — Married vs Unmarried')
axes[0].set_xlabel('Purchase Amount ($)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

sns.boxplot(data=df, x='Marital_Status', y='Purchase', palette=['teal', 'coral'], ax=axes[1])
axes[1].set_title('Purchase Box Plot — Marital Status')
axes[1].set_xticklabels(['Unmarried', 'Married'])
axes[1].set_ylabel('Purchase Amount ($)')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '11_marital_spending.png'), bbox_inches='tight')
plt.show()

# %%
# ── 5.4.3  Confidence Intervals — Marital Status ───────────────────────────
print("=" * 80)
print("   CONFIDENCE INTERVALS — UNMARRIED")
print("=" * 80)
ci_unmarried = compute_confidence_interval(unmarried_purchases)
print(ci_unmarried.to_string(index=False))

print("\n" + "=" * 80)
print("   CONFIDENCE INTERVALS — MARRIED")
print("=" * 80)
ci_married = compute_confidence_interval(married_purchases)
print(ci_married.to_string(index=False))

# %%
# ── 5.4.4  Overlap Analysis — Married vs Unmarried ─────────────────────────
print("=" * 90)
print("   CI OVERLAP ANALYSIS — UNMARRIED vs MARRIED")
print("=" * 90)

overlap_marital = []
for conf in ['90%', '95%', '99%']:
    for n in [30, 100, 500, 1000]:
        u_row = ci_unmarried[(ci_unmarried['Confidence Level'] == conf) &
                              (ci_unmarried['Sample Size'] == n)].iloc[0]
        m_row = ci_married[(ci_married['Confidence Level'] == conf) &
                            (ci_married['Sample Size'] == n)].iloc[0]

        overlap = u_row['CI Lower'] <= m_row['CI Upper'] and m_row['CI Lower'] <= u_row['CI Upper']
        overlap_amt = 0
        if overlap:
            overlap_amt = min(u_row['CI Upper'], m_row['CI Upper']) - max(u_row['CI Lower'], m_row['CI Lower'])

        overlap_marital.append({
            'Confidence': conf,
            'Sample Size': n,
            'Unmarried CI': f"[{u_row['CI Lower']:,.0f}, {u_row['CI Upper']:,.0f}]",
            'Married CI': f"[{m_row['CI Lower']:,.0f}, {m_row['CI Upper']:,.0f}]",
            'Overlap?': '✅ Yes' if overlap else '❌ No',
        })

print(pd.DataFrame(overlap_marital).to_string(index=False))

# %%
# ── 5.4.5  CI Visualisation — Marital Status ───────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

ci_u_95 = ci_unmarried[ci_unmarried['Confidence Level'] == '95%']
ci_m_95 = ci_married[ci_married['Confidence Level'] == '95%']

y_pos = 0
y_labels = []
for u_row, m_row in zip(ci_u_95.itertuples(), ci_m_95.itertuples()):
    ax.barh(y_pos + 0.2, u_row._8 - u_row._7, left=u_row._7, height=0.35,
            color='teal', alpha=0.7, edgecolor='black', label='Unmarried' if y_pos == 0 else '')
    ax.barh(y_pos - 0.2, m_row._8 - m_row._7, left=m_row._7, height=0.35,
            color='coral', alpha=0.7, edgecolor='black', label='Married' if y_pos == 0 else '')
    ax.plot(u_row._3, y_pos + 0.2, 'D', color='darkgreen', markersize=8)
    ax.plot(m_row._3, y_pos - 0.2, 'D', color='darkred', markersize=8)
    y_labels.append(f'n = {int(u_row._1)}')
    y_pos += 1.2

ax.set_yticks([i * 1.2 for i in range(len(y_labels))])
ax.set_yticklabels(y_labels, fontsize=12)
ax.set_xlabel('Purchase Amount ($)', fontsize=12)
ax.set_title('95% Confidence Intervals — Unmarried vs Married (by Sample Size)', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '12_ci_overlap_marital.png'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 📌 Inference — Marital Status
#
# - The **average spending** for married and unmarried customers appears very
#   similar.
# - CIs show **substantial overlap** across all sample sizes and confidence
#   levels, suggesting that marital status **does not significantly affect**
#   purchase amount.
# - **Business implication**: Walmart need not differentiate Black Friday
#   campaigns based on marital status.

# %% [markdown]
# ---
# ## 5.5 Age Group Analysis *(10 pts)*

# %%
# ── 5.5.1  Descriptive Stats — Age Groups ──────────────────────────────────
age_stats = df.groupby('Age')['Purchase'].agg(['mean', 'median', 'std', 'count'])
age_stats.columns = ['Mean ($)', 'Median ($)', 'Std Dev ($)', 'Count']
age_stats = age_stats.reindex(order_age)
print("=" * 70)
print("   SPENDING BY AGE GROUP")
print("=" * 70)
print(age_stats.round(2))

# %%
# ── 5.5.2  Visualise Age Group Spending ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

age_means = df.groupby('Age')['Purchase'].mean().reindex(order_age)
bars = axes[0].bar(order_age, age_means, color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(order_age))),
                   edgecolor='black')
axes[0].set_title('Average Purchase by Age Group')
axes[0].set_xlabel('Age Group')
axes[0].set_ylabel('Average Purchase ($)')
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                 f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

sns.violinplot(data=df, x='Age', y='Purchase', order=order_age,
               palette='coolwarm', ax=axes[1], inner='box')
axes[1].set_title('Purchase Distribution by Age (Violin Plot)')
axes[1].set_xlabel('Age Group')
axes[1].set_ylabel('Purchase Amount ($)')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '13_age_spending.png'), bbox_inches='tight')
plt.show()

# %%
# ── 5.5.3  Consolidate Age Bins for CI Analysis ────────────────────────────
# Group into broader bins: 0-17, 18-25, 26-35, 36-50, 51+
age_bin_map = {
    '0-17': '0-17',
    '18-25': '18-25',
    '26-35': '26-35',
    '36-45': '36-50',
    '46-50': '36-50',
    '51-55': '51+',
    '55+': '51+',
}
df['Age_Broad'] = df['Age'].map(age_bin_map)
broad_age_order = ['0-17', '18-25', '26-35', '36-50', '51+']

print("Broad Age Bins Transaction Count:")
print(df['Age_Broad'].value_counts().reindex(broad_age_order))

# %%
# ── 5.5.4  Confidence Intervals — Age Groups ──────────────────────────────
ci_age_results = {}
for age_grp in broad_age_order:
    age_data = df[df['Age_Broad'] == age_grp]['Purchase'].values
    ci_age_results[age_grp] = compute_confidence_interval(age_data)

# Display CIs for each group
for age_grp in broad_age_order:
    print(f"\n{'='*80}")
    print(f"   CONFIDENCE INTERVALS — AGE GROUP: {age_grp}")
    print(f"{'='*80}")
    print(ci_age_results[age_grp].to_string(index=False))

# %%
# ── 5.5.5  Visualise CIs — Age Groups at 95% ──────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))

palette = plt.cm.Set2(np.linspace(0, 1, len(broad_age_order)))
y_pos = 0
y_labels_list = []
legend_handles = []

for grp_idx, age_grp in enumerate(broad_age_order):
    ci_sub = ci_age_results[age_grp]
    ci_95 = ci_sub[ci_sub['Confidence Level'] == '95%']

    for _, row in ci_95.iterrows():
        bar = ax.barh(y_pos, row['CI Upper'] - row['CI Lower'], left=row['CI Lower'],
                      height=0.6, color=palette[grp_idx], alpha=0.7, edgecolor='black')
        ax.plot(row['Population Mean'], y_pos, 'D', color='black', markersize=7, zorder=5)
        y_labels_list.append(f'{age_grp} (n={int(row["Sample Size"])})')
        y_pos += 1

    if grp_idx < len(broad_age_order):
        import matplotlib.patches as mpatches
        legend_handles.append(mpatches.Patch(color=palette[grp_idx], label=age_grp))
    y_pos += 0.5  # gap between groups

ax.set_yticks(range(len(y_labels_list)))
ax.set_yticklabels(y_labels_list, fontsize=9)
ax.set_xlabel('Purchase Amount ($)', fontsize=12)
ax.set_title('95% Confidence Intervals — All Age Groups (by Sample Size)', fontsize=14)
ax.legend(handles=legend_handles, loc='upper right', fontsize=10, title='Age Group')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '14_ci_age_groups.png'), bbox_inches='tight')
plt.show()

# %%
# ── 5.5.6  Age Group Overlap Summary ───────────────────────────────────────
print("=" * 90)
print("   PAIRWISE CI OVERLAP — AGE GROUPS (95% CI, n=1000)")
print("=" * 90)

from itertools import combinations

age_pair_overlaps = []
for g1, g2 in combinations(broad_age_order, 2):
    ci1 = ci_age_results[g1]
    ci1_row = ci1[(ci1['Confidence Level'] == '95%') & (ci1['Sample Size'] == 1000)].iloc[0]
    ci2 = ci_age_results[g2]
    ci2_row = ci2[(ci2['Confidence Level'] == '95%') & (ci2['Sample Size'] == 1000)].iloc[0]

    overlap = ci1_row['CI Lower'] <= ci2_row['CI Upper'] and ci2_row['CI Lower'] <= ci1_row['CI Upper']

    age_pair_overlaps.append({
        'Group 1': g1,
        'Group 2': g2,
        f'Mean {g1}': f"${ci1_row['Population Mean']:,.0f}",
        f'Mean {g2}': f"${ci2_row['Population Mean']:,.0f}",
        'Overlap?': '✅ Yes' if overlap else '❌ No',
        'Diff ($)': f"${abs(ci1_row['Population Mean'] - ci2_row['Population Mean']):,.0f}",
    })

print(pd.DataFrame(age_pair_overlaps).to_string(index=False))

# %% [markdown]
# ### 📌 Inference — Age Groups
#
# - Some age groups (e.g., **51+** and **36-50**) show higher average spending.
# - With **large sample sizes (n ≥ 500)**, some pairs show **non-overlapping CIs**,
#   suggesting statistically significant differences in spending.
# - The **0-17** group and **26-35** group may show distinct spending patterns
#   compared to older demographics.
# - **Business implication**: Walmart should consider **age-targeted promotions** —
#   offering different product bundles or discount levels for high-spending vs
#   lower-spending age groups.

# %% [markdown]
# ---
# # 6. Final Insights *(10 pts)*
#
# ## 📊 Distribution Observations
# - The **Purchase amount** distribution is **right-skewed** (positively skewed),
#   meaning most transactions are moderate, with a long tail of high-value
#   purchases.
# - Despite this non-normal population distribution, the **CLT** ensures that
#   the **distribution of sample means becomes approximately normal** as sample
#   size increases — confirmed by our bootstrap simulations.
#
# ## 🔬 CLT Observations
# - At **n = 30**, the sampling distribution already starts resembling a bell
#   curve, but CIs are wide.
# - At **n = 100**, the normal shape is clearly visible.
# - At **n ≥ 500**, the distribution is tightly normal and CIs are narrow
#   enough for meaningful comparisons.
#
# ## 🧪 Key Statistical Findings
#
# | Analysis | Finding |
# |---|---|
# | **Male vs Female** | Males spend slightly more per transaction on average. CIs may overlap at small samples but diverge at larger sizes. |
# | **Married vs Unmarried** | No significant difference in spending — CIs overlap substantially. |
# | **Age Groups** | Older groups (36-50, 51+) tend to spend more. Significant differences emerge at larger samples. |
# | **CLT Validation** | The sampling distribution of means converges to normality regardless of the skewed population distribution. |

# %% [markdown]
# ---
# # 7. Recommendations *(10 pts)*
#
# Based on the statistical analysis, here are actionable recommendations for
# Walmart's Black Friday strategy:
#
# ## 🎯 Gender-Based Strategy
# 1. **Males are the primary spenders** — allocate higher marketing budget toward
#    male-centric product categories (electronics, gadgets, tools).
# 2. **Female spending uplift opportunity** — introduce targeted campaigns
#    (bundle deals, loyalty rewards) to increase female per-transaction spend.
# 3. If CIs overlap, apply a **gender-neutral campaign** as default.
#
# ## 💍 Marital Status Strategy
# 4. **No need to segment by marital status** — spending patterns are
#    statistically similar. Keep campaigns unified.
#
# ## 📅 Age-Based Strategy
# 5. **26-35 age group** dominates volume — this is the *core demographic*.
#    Ensure product availability and aggressive promotions for this segment.
# 6. **51+ and 36-50 groups** show higher per-transaction spending — create
#    premium product bundles and upselling opportunities for these segments.
# 7. **0-17 age group** spends least — introduce entry-level products and
#    gamified shopping experiences to build brand loyalty early.
#
# ## 🏙️ City-Based Strategy
# 8. **City B** drives the most transactions — prioritise inventory and
#    promotional events in Tier-2 cities.
# 9. **City C** has untapped potential — invest in awareness campaigns.
#
# ## 📈 Data & Analytics
# 10. **Collect more granular data** — time-of-day, product sub-categories,
#     payment method — to enable deeper segmentation.
# 11. **A/B test** the recommendations above with controlled experiments
#     before full-scale rollout.
# 12. **Repeat this analysis quarterly** to track shifts in consumer behaviour.

# %%
# ── Final Summary Table ─────────────────────────────────────────────────────
summary_data = {
    'Segment': ['Male', 'Female', 'Unmarried', 'Married',
                '0-17', '18-25', '26-35', '36-50', '51+'],
    'Avg Purchase ($)': [
        round(male_mean, 2),
        round(female_mean, 2),
        round(df[df['Marital_Status'] == 0]['Purchase'].mean(), 2),
        round(df[df['Marital_Status'] == 1]['Purchase'].mean(), 2),
    ] + [round(df[df['Age_Broad'] == a]['Purchase'].mean(), 2) for a in broad_age_order],
    'Transaction Count': [
        len(male_purchases),
        len(female_purchases),
        len(unmarried_purchases),
        len(married_purchases),
    ] + [len(df[df['Age_Broad'] == a]) for a in broad_age_order],
}
summary_df = pd.DataFrame(summary_data)
print("=" * 65)
print("   FINAL SUMMARY — ALL SEGMENTS")
print("=" * 65)
print(summary_df.to_string(index=False))

# %%
# ── Save Summary as CSV ─────────────────────────────────────────────────────
summary_df.to_csv(os.path.join(IMG_DIR, 'segment_summary.csv'), index=False)
print(f"\n✅ Summary saved to {os.path.join(IMG_DIR, 'segment_summary.csv')}")
print("✅ All visualisations saved to the 'images/' directory.")
print("\n🏁 Analysis Complete — Walmart CLT & Confidence Interval Case Study")
