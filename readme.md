# 🧠 Hate Speech Data Analysis & Preprocessing (HateXplain Dataset)

This project performs a **comprehensive data analysis, cleaning, encoding, and visualization** of the HateXplain dataset to prepare it for machine learning tasks.

It demonstrates a full **data preprocessing pipeline**, including data quality assessment, feature analysis, and visualization.

---

## 🚀 Project Overview

The goal of this project is to analyze and prepare the **HateXplain dataset** for downstream machine learning tasks such as hate speech classification.

The dataset contains annotated social media posts labeled as:
- **Hate Speech**
- **Offensive**
- **Normal**

---

## 📊 Dataset Details

- Dataset: HateXplain
- Total Records: **60,444**
- Columns:
  - `post_id`
  - `annotator_id`
  - `label`
  - `target`
  - `post_tokens`

---

## 🧹 Data Processing Pipeline

### 1️⃣ Data Loading & Exploration
- Loaded dataset using Pandas
- Analyzed dataset structure, size, and data types

### 2️⃣ Data Quality Assessment
- Identified missing values (~7% of total data)
- Found **35% missing values in `target` column**
- Checked for duplicate records (none found)

### 3️⃣ Data Cleaning
- Filled missing `target` values with `"Unknown"`
- Removed invalid or incomplete entries
- Reset dataset index after cleaning

### 4️⃣ Feature Analysis
- Variance analysis for numerical features
- Identified low-variance features
- Explored categorical distributions

### 5️⃣ Encoding
- Applied **Label Encoding** on:
  - `label` (hate, offensive, normal)
  - `target` (multiple demographic groups)

---

## 📈 Visualizations

The project generates multiple visual insights:

### 📊 1. Label Distribution
- Shows class imbalance across hate speech categories

### 📊 2. Target Group Distribution
- Top affected demographic groups in dataset

### 📊 3. Text Length Distribution
- Distribution of post lengths with mean & median

### 📊 4. Correlation Heatmap
- Relationships between numerical features

Generated files:
chart1_label_distribution.png
chart2_target_distribution.png
chart3_text_length_distribution.png
chart4_correlation_matrix.png


---

## ⚙️ Technologies Used

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn (Label Encoding)**

---

## 📂 Project Structure
code.py # Main preprocessing & analysis script
├── hateXplain.csv # Dataset
├── cleaned_hateXplain_data.csv # Cleaned dataset output
├── output.txt # Execution results
├── charts/
│ ├── chart1_label_distribution.png
│ ├── chart2_target_distribution.png
│ ├── chart3_text_length_distribution.png
│ └── chart4_correlation_matrix.png
├── README.md


## ▶️ How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
