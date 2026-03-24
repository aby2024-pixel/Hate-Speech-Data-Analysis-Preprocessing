import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("MILESTONE 1: COMPREHENSIVE DATA ANALYSIS - HATEXPLAIN DATASET")
print("=" * 70)

# STEP 1: DATASET LOADING AND READING
print("STEP 1: LOADING HATEXPLAIN DATASET")
print("=" * 60)

# Load the hateXplain dataset
df = pd.read_csv('hateXplain.csv')

print(f"Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 records:")
print(df.head())

print(f"\nDataset Info:")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\nColumn Data Types:")
for col in df.columns:
    print(f"{col:15}: {df[col].dtype}")

# STEP 2: DATA QUALITY ASSESSMENT - MISSING VALUES, NULL VARIABLES
print("\nSTEP 2: DATA QUALITY ASSESSMENT")
print("=" * 60)

print("Missing Values Analysis:")
missing_summary = df.isnull().sum()
total_cells = len(df) * len(df.columns)
total_missing = missing_summary.sum()

print(f"Total missing values: {total_missing}")
print(f"Percentage of missing data: {(total_missing / total_cells) * 100:.2f}%")

for col, missing_count in missing_summary.items():
    if missing_count > 0:
        missing_percent = (missing_count / len(df)) * 100
        print(f"{col:15}: {missing_count:3d} missing ({missing_percent:.1f}%)")
    else:
        print(f"{col:15}: {missing_count:3d} missing (0.0%)")

print(f"\nNull Variable Analysis:")
null_columns = []
for col in df.columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        null_columns.append(col)
        print(f"Column '{col}' has {null_count} null values")

if not null_columns:
    print("No columns with null values found")

# STEP 3: DUPLICATE DETECTION AND HANDLING
print(f"\nSTEP 3: DUPLICATE ANALYSIS")
print("=" * 60)

duplicates = df.duplicated().sum()
print(f"Total duplicate records: {duplicates}")
print(f"Percentage of duplicates: {(duplicates / len(df)) * 100:.2f}%")

if duplicates > 0:
    print("Sample duplicate rows:")
    duplicate_rows = df[df.duplicated(keep=False)]
    print(duplicate_rows.head())

# STEP 4: DATA CLEANING PROCESS - FIXED VERSION
print("\nSTEP 4: DATA CLEANING PROCESS")
print("=" * 60)

df_clean = df.copy()
rows_before_cleaning = len(df_clean)

print("Cleaning Steps Applied:")

# Step 1: Handle missing values
print("1. Handling Missing Values:")
rows_after_missing = rows_before_cleaning

# Handle missing target values
if 'target' in df_clean.columns and df_clean['target'].isnull().sum() > 0:
    missing_targets = df_clean['target'].isnull().sum()
    df_clean['target'].fillna('Unknown', inplace=True)
    print(f"   - Filled {missing_targets} missing target values with 'Unknown'")

# Handle missing post_tokens by removing rows
if 'post_tokens' in df_clean.columns and df_clean['post_tokens'].isnull().sum() > 0:
    missing_tokens_before = df_clean['post_tokens'].isnull().sum()
    df_clean = df_clean.dropna(subset=['post_tokens'])
    rows_after_missing = len(df_clean)
    removed_tokens = rows_before_cleaning - rows_after_missing
    print(f"   - Removed {removed_tokens} rows with missing post_tokens")
else:
    print(f"   - No missing post_tokens found")

# Step 2: Remove duplicates
print("2. Removing Duplicates:")
duplicates_before = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
rows_after_dedup = len(df_clean)
removed_duplicates = rows_after_missing - rows_after_dedup
print(f"   - Removed {removed_duplicates} duplicate rows")

# Reset index after cleaning
df_clean = df_clean.reset_index(drop=True)

print(f"\nCleaning Summary:")
print(f"   - Original rows: {rows_before_cleaning}")
print(f"   - After missing value removal: {rows_after_missing}")
print(f"   - After duplicate removal: {rows_after_dedup}")
print(f"   - Total rows removed: {rows_before_cleaning - rows_after_dedup}")
print(f"   - Retention rate: {(rows_after_dedup / rows_before_cleaning) * 100:.1f}%")

# STEP 5: VARIANCE ANALYSIS
print("\nSTEP 5: VARIANCE ANALYSIS")
print("=" * 60)

# Calculate variance for numerical columns
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

print("Variance Analysis for Numerical Columns:")
variance_data = {}
for col in numerical_cols:
    if len(df_clean[col].dropna()) > 1:  # Ensure we have enough data points
        variance = df_clean[col].var()
        std_dev = df_clean[col].std()
        variance_data[col] = {'variance': variance, 'std_dev': std_dev}
        print(f"{col:20}: Variance = {variance:.6f}, Std Dev = {std_dev:.6f}")

# Identify low variance features
low_variance_threshold = 0.01
print(f"\nLow Variance Features (variance < {low_variance_threshold}):")
low_variance_features = []
for col, stats in variance_data.items():
    if stats['variance'] < low_variance_threshold:
        low_variance_features.append(col)
        print(f"   - {col}: Variance = {stats['variance']:.6f}")

if not low_variance_features:
    print("   - No low variance features found")

# STEP 6: COLUMN ENUMERATION AND ENCODING
print("\nSTEP 6: COLUMN ENUMERATION AND ENCODING")
print("=" * 60)

# Identify categorical columns
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns found: {categorical_cols}")

# Encode main categorical variables
encoders = {}
encoding_results = {}

# Encode labels if present
if 'label' in df_clean.columns:
    le_label = LabelEncoder()
    df_clean['label_encoded'] = le_label.fit_transform(df_clean['label'])
    encoders['label'] = le_label
    encoding_results['label'] = dict(zip(le_label.classes_, le_label.transform(le_label.classes_)))
    print(f"\nLabel Encoding Results:")
    for original, encoded in encoding_results['label'].items():
        count = (df_clean['label'] == original).sum()
        print(f"   {original:12} -> {encoded} ({count} instances)")

# Encode targets if present
if 'target' in df_clean.columns:
    le_target = LabelEncoder()
    df_clean['target_encoded'] = le_target.fit_transform(df_clean['target'])
    encoders['target'] = le_target
    encoding_results['target'] = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
    print(f"\nTarget Encoding Results:")
    for original, encoded in encoding_results['target'].items():
        count = (df_clean['target'] == original).sum()
        print(f"   {original:12} -> {encoded} ({count} instances)")

# STEP 7: COMPREHENSIVE STATISTICAL ANALYSIS
print("\nSTEP 7: STATISTICAL SUMMARY")
print("=" * 60)

# Update numerical columns list to include encoded variables
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

print("Descriptive Statistics for Numerical Variables:")
if numerical_cols:
    desc_stats = df_clean[numerical_cols].describe()
    print(desc_stats)
else:
    print("No numerical columns found")

print(f"\nCategorical Variables Summary:")
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    print(f"\n{col} distribution:")
    value_counts = df_clean[col].value_counts()
    for value, count in value_counts.head(10).items():
        percentage = (count / len(df_clean)) * 100
        print(f"   {str(value)[:15]:15}: {count:5d} ({percentage:.1f}%)")

# STEP 8: CORRELATION ANALYSIS
print(f"\nSTEP 8: CORRELATION ANALYSIS")
print("=" * 60)

if len(numerical_cols) > 1:
    correlation_matrix = df_clean[numerical_cols].corr()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Identify high correlations
    high_corr_threshold = 0.7
    print(f"\nHigh Correlations (absolute value > {high_corr_threshold}):")
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > high_corr_threshold:
                pair = (correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val)
                high_corr_pairs.append(pair)
                print(f"   {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
    
    if not high_corr_pairs:
        print("   No high correlations found")
else:
    print("Not enough numerical columns for correlation analysis")

print("\n" + "=" * 70)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"Final Dataset Shape: {df_clean.shape}")
print("Ready for Visualization and Machine Learning")

# Save cleaned dataset
df_clean.to_csv('cleaned_hateXplain_data.csv', index=False)
print("Dataset saved as 'cleaned_hateXplain_data.csv'")

# VISUALIZATION CODE (40% of grade)
print("\nSTEP 9: GENERATING VISUALIZATIONS")
print("=" * 60)

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Chart 1: Label Distribution
if 'label' in df_clean.columns:
    plt.figure(figsize=(12, 6))
    label_counts = df_clean['label'].value_counts()
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = plt.bar(label_counts.index, label_counts.values, color=colors[:len(label_counts)])
    
    # Add value labels on bars
    total = len(df_clean)
    for i, (bar, count) in enumerate(zip(bars, label_counts.values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(label_counts.values)*0.01,
                f'{count:,}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.title('Distribution of Hate Speech Labels in Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Label Categories', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Instances', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('chart1_label_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Chart 1: Label Distribution saved")

# Chart 2: Target Group Distribution  
if 'target' in df_clean.columns:
    plt.figure(figsize=(14, 8))
    target_counts = df_clean['target'].value_counts().head(10)  # Top 10 targets
    colors = plt.cm.viridis(np.linspace(0, 1, len(target_counts)))
    bars = plt.barh(range(len(target_counts)), target_counts.values, color=colors)
    plt.yticks(range(len(target_counts)), [str(x)[:20] for x in target_counts.index])
    
    plt.title('Top 10 Target Groups in Hate Speech Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Instances', fontsize=12, fontweight='bold')
    plt.ylabel('Target Groups', fontsize=12, fontweight='bold')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, target_counts.values)):
        width = bar.get_width()
        plt.text(width + max(target_counts.values)*0.01, bar.get_y() + bar.get_height()/2,
                f'{value:,}', va='center', ha='left', fontweight='bold')
    
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('chart2_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Chart 2: Target Distribution saved")

# Chart 3: Text Length Distribution (if available)
text_length_col = None
for col in df_clean.columns:
    if 'length' in col.lower() or 'len' in col.lower():
        text_length_col = col
        break

if text_length_col and df_clean[text_length_col].dtype in ['int64', 'float64']:
    plt.figure(figsize=(12, 6))
    data = df_clean[text_length_col].dropna()
    
    # Create histogram
    n, bins, patches = plt.hist(data, bins=30, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add statistical lines
    mean_val = data.mean()
    median_val = data.median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    
    plt.title('Distribution of Text Lengths in Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Text Length (characters)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('chart3_text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Chart 3: Text Length Distribution saved")
else:
    # Create a sample text length analysis
    if 'post_tokens' in df_clean.columns:
        df_clean['text_length'] = df_clean['post_tokens'].str.len()
        plt.figure(figsize=(12, 6))
        data = df_clean['text_length'].dropna()
        
        n, bins, patches = plt.hist(data, bins=30, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
        mean_val = data.mean()
        median_val = data.median()
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
        
        plt.title('Distribution of Text Lengths in Dataset', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Text Length (characters)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig('chart3_text_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Chart 3: Text Length Distribution saved")

# Chart 4: Correlation Heatmap
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df_clean[numerical_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', 
                cmap='RdBu_r', center=0, square=True, cbar_kws={'shrink': 0.8},
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('chart4_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Chart 4: Correlation Matrix saved")
else:
    print("Not enough numerical columns for correlation matrix")

print("\n" + "=" * 70)
print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("Files generated:")
print("- cleaned_hateXplain_data.csv")
print("- chart1_label_distribution.png")  
print("- chart2_target_distribution.png")
print("- chart3_text_length_distribution.png")
print("- chart4_correlation_matrix.png")
print("\nMilestone 1 Analysis Complete! ✓")