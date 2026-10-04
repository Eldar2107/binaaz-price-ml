import pandas as pd

PATH = "data/bina_az_sale.csv"   # fayl adı fərqlidirsə dəyiş

df = pd.read_csv(PATH, encoding="utf-8", low_memory=False)

print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS + DTYPES + MISSING ===")
info = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_missing": df.isna().sum(),
    "pct_missing": (df.isna().mean() * 100).round(1),
    "n_unique": df.nunique(),
})
print(info.to_string())

print("\n=== FIRST 5 ROWS ===")
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(df.head())

print("\n=== DUPLICATES ===")
print("fully duplicated rows:", df.duplicated().sum())

print("\n=== PRICE ===")
if "price" in df.columns:
    p = pd.to_numeric(df["price"], errors="coerce")
    print("non-numeric price values:", p.isna().sum() - df["price"].isna().sum())
    print(p.describe(percentiles=[.01, .05, .5, .95, .99]).to_string())
    print("price <= 0:", (p <= 0).sum())
else:
    print("No column named 'price' - check the names above.")

print("\n=== LEAKAGE CANDIDATES ===")
keys = ("price", "unit", "total", "address", "owner", "shop")
print([c for c in df.columns if any(k in str(c).lower() for k in keys)])

print("\n=== LOW-CARDINALITY CATEGORICALS (<= 15 values) ===")
for c in df.columns:
    if df[c].nunique() <= 15:
        print(f"\n{c}:")
        print(df[c].value_counts(dropna=False).head(15).to_string())