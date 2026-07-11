import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy import stats
except ImportError:
    stats = None


DATASET_CANDIDATES = [
    "flipkart.csv",
    "flipkart_data.csv",
    "flipkart_products.csv",
    "data.csv",
    "dataset.csv",
]


def load_data(base_dir: Path) -> pd.DataFrame:
    for name in DATASET_CANDIDATES:
        path = base_dir / name
        if path.exists():
            print(f"Loading dataset from {path}")
            return pd.read_csv(path)

    print("No dataset file found. Using a built-in sample Flipkart-style dataset.")
    return create_sample_dataset()


def create_sample_dataset() -> pd.DataFrame:
    data = {
        "product_name": [
            "Redmi Note 13 Pro",
            "Samsung Galaxy S24",
            "OnePlus Nord 4",
            "Sony WH-1000XM5",
            "Dell Inspiron 15",
            "Apple iPhone 15",
            "Noise Smart Watch",
            "Canon EOS 3000D",
            "boAt Rockerz 450",
            "Philips Trimmer",
            "Lenovo IdeaPad Slim 3",
            "Realme Buds Air 6",
            "Puma Running Shoes",
            "Levi's Casual Shirt",
            "Adidas Backpack",
        ],
        "category": [
            "Mobiles",
            "Mobiles",
            "Mobiles",
            "Audio",
            "Laptops",
            "Mobiles",
            "Wearables",
            "Cameras",
            "Audio",
            "Personal Care",
            "Laptops",
            "Audio",
            "Footwear",
            "Fashion",
            "Accessories",
        ],
        "brand": [
            "Redmi",
            "Samsung",
            "OnePlus",
            "Sony",
            "Dell",
            "Apple",
            "Noise",
            "Canon",
            "boAt",
            "Philips",
            "Lenovo",
            "Realme",
            "Puma",
            "Levi's",
            "Adidas",
        ],
        "price": [17999, 79999, 24999, 29990, 54990, 79999, 2999, 32999, 1499, 1999, 38990, 2999, 4999, 1999, 3499],
        "original_price": [21999, 89999, 29999, 34990, 59990, 89999, 3999, 36999, 1999, 2499, 42990, 3999, 5999, 2499, 4999],
        "discount_percent": [18, 11, 17, 14, 8, 11, 25, 11, 25, 20, 9, 25, 17, 20, 30],
        "rating": [4.4, 4.8, 4.2, 4.7, 4.1, 4.9, 4.3, 4.5, 4.2, 4.0, 4.0, 4.4, 4.3, 4.1, 4.2],
        "reviews": [12560, 9830, 4300, 6720, 1640, 11800, 5210, 1160, 4780, 2140, 2750, 3140, 1680, 1320, 1900],
        "stock": [48, 32, 55, 21, 10, 12, 87, 6, 91, 64, 14, 77, 43, 68, 35],
    }
    df = pd.DataFrame(data)
    df["discount_amount"] = df["original_price"] - df["price"]
    df["rating_bucket"] = np.where(df["rating"] >= 4.5, "High", "Medium")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    for col in ["price", "original_price", "discount_percent", "rating", "reviews", "stock"]:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    if "product_name" in clean_df.columns:
        clean_df["product_name"] = clean_df["product_name"].astype(str).str.strip()
    if "category" in clean_df.columns:
        clean_df["category"] = clean_df["category"].fillna("Unknown")
    if "brand" in clean_df.columns:
        clean_df["brand"] = clean_df["brand"].fillna("Unknown")

    clean_df = clean_df.drop_duplicates()
    return clean_df


def detect_outliers(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)

    print("\n=== Meaningful Questions ===")
    questions = [
        "Which product categories have the highest number of listings?",
        "How does price vary across categories and brands?",
        "Do higher-rated products also command higher prices?",
        "Are there any unusual price or discount outliers?",
    ]
    for q in questions:
        print(f"- {q}")

    print("\n=== Data Structure ===")
    print(df.info())
    print("\nSample rows:")
    print(df.head().to_string(index=False))

    print("\n=== Summary Statistics ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(df[numeric_cols].describe().transpose().to_string())

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== Duplicate Rows ===")
    print(f"Duplicates: {df.duplicated().sum()}")

    print("\n=== Category Distribution ===")
    print(df["category"].value_counts().to_string())

    print("\n=== Top Brands by Count ===")
    if "brand" in df.columns:
        print(df["brand"].value_counts().head(10).to_string())

    if "price" in df.columns:
        print("\n=== Price by Category ===")
        print(df.groupby("category")["price"].agg(["count", "mean", "median", "max"]).round(2).to_string())

    if "rating" in df.columns and "price" in df.columns:
        print("\n=== Correlation ===")
        corr = df[["price", "rating", "reviews", "discount_percent", "stock"]].corr(numeric_only=True)
        print(corr.round(2).to_string())

    if "price" in df.columns and "category" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="category", y="price")
        plt.xticks(rotation=45)
        plt.title("Price Distribution by Category")
        plt.tight_layout()
        plt.savefig(output_dir / "price_by_category.png", dpi=150)
        plt.close()

    if "category" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="category", order=df["category"].value_counts().index)
        plt.xticks(rotation=45)
        plt.title("Category Counts")
        plt.tight_layout()
        plt.savefig(output_dir / "category_counts.png", dpi=150)
        plt.close()

    if "price" in df.columns and "rating" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="rating", y="price", hue="category", s=90)
        plt.title("Price vs Rating")
        plt.tight_layout()
        plt.savefig(output_dir / "price_vs_rating.png", dpi=150)
        plt.close()

    if numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.title("Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
        plt.close()

    if "price" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["price"], kde=True, bins=10)
        plt.title("Price Distribution")
        plt.tight_layout()
        plt.savefig(output_dir / "price_distribution.png", dpi=150)
        plt.close()

    print("\n=== Outliers ===")
    if "price" in df.columns:
        outlier_mask = detect_outliers(df["price"])
        print(f"Price outliers: {df.loc[outlier_mask, ['product_name', 'category', 'price']].to_string(index=False)}")

    if "rating" in df.columns and "price" in df.columns:
        high_rating = df[df["rating"] >= 4.5]["price"]
        low_rating = df[df["rating"] < 4.5]["price"]
        print("\n=== Hypothesis Test ===")
        if not high_rating.empty and not low_rating.empty and stats is not None:
            t_stat, p_value = stats.ttest_ind(high_rating, low_rating, equal_var=False)
            print(f"H0: Mean price is the same for high- and low-rated products")
            print(f"Welch t-test p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("Conclusion: There is evidence of a price difference between high- and low-rated products.")
            else:
                print("Conclusion: No strong evidence of a price difference between high- and low-rated products.")
        else:
            print("SciPy is unavailable, so the statistical hypothesis test was skipped.")

    print("\n=== Key Findings ===")
    if "category" in df.columns and "price" in df.columns:
        top_category = df.groupby("category")["price"].mean().idxmax()
        print(f"Highest average price category: {top_category}")
    if "rating" in df.columns and "price" in df.columns:
        best_rating = df.loc[df["rating"].idxmax(), "product_name"]
        print(f"Highest-rated product: {best_rating}")

    print("\n=== Potential Data Issues ===")
    if df.isnull().sum().sum() > 0:
        print("- Missing values were found and should be handled before modeling.")
    if "price" in df.columns and (df["price"] <= 0).any():
        print("- Some prices are non-positive and may need cleaning.")
    if "discount_percent" in df.columns and (df["discount_percent"] < 0).any():
        print("- Negative discount values were detected.")
    if "rating" in df.columns and (df["rating"] > 5).any():
        print("- Ratings above 5 were detected; these may be invalid.")

    report_path = output_dir / "flipkart_eda_report.txt"
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("Flipkart EDA Report\n")
        report_file.write("==================\n")
        report_file.write(f"Rows: {len(df)}\n")
        report_file.write(f"Columns: {list(df.columns)}\n\n")
        report_file.write("Key questions:\n")
        for q in questions:
            report_file.write(f"- {q}\n")
        report_file.write("\nSummary:\n")
        report_file.write(f"- Missing values: {df.isnull().sum().sum()}\n")
        report_file.write(f"- Duplicates: {df.duplicated().sum()}\n")

    print(f"\nEDA completed. Outputs saved in {output_dir}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    df = load_data(base_dir)
    df = clean_data(df)
    run_eda(df, base_dir / "eda_outputs")


if __name__ == "__main__":
    main()
