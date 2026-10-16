# Data

The **bina.az sale** dataset is **not committed** to the repo (see `.gitignore`).
Each team member downloads it locally.

## Where to get it
Kaggle: **"binaaz-sale-project"** by `sehriyarmemmedli`
<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>

## File Specifications and Provenance
- **Filename:** `bina_az_sale.csv`
- **File size:** 141,882,994 bytes
- **SHA-256 Checksum:** `b6a4c67e6c4e0b0439247169bbd1dbf43501a357eb4659f1eb175b5b4e72aec2`
- **Download date:** October 2026

## How to place it
1. Download the dataset archive from Kaggle.
2. Extract it so the CSV lives **directly in this folder**, e.g.:
   ```
   data/bina_az_sale.csv
   ```
3. If your filename differs, set the path once at the top of
   `src/data_prep.py` (the `DATA_PATH` constant) — do not scatter paths
   across the codebase.

## Integrity Verification
To verify the file integrity on Windows PowerShell:
```powershell
Get-FileHash -Algorithm SHA256 .\data\bina_az_sale.csv
```
Expected hash:
`B6A4C67E6C4E0B0439247169BBD1DBF43501A357EB4659F1EB175B5B4E72AEC2`

## Notes / gotchas
- Column names are in **Azerbaijani** (e.g. `Sahə`, `Otaq sayı`, `Mərtəbə`,
  `Təmir`, `Kateqoriya`, …). Keep the UTF-8 encoding when you read the file.
- **Leakage warning:** any column derived from the target `price`
  (e.g. a per-m² unit price, or a precomputed total like `unit_price`, `total_price`) must be **dropped**
  before training. These columns are explicitly removed during cleaning in `src/data_prep.py`.
- Record the dataset version/date you downloaded, for reproducibility.
