# User Guide

## 0) Core idea — choosing the VALUE column (this project’s key control)
`run_all_test` lets you **manually tell the pipeline** which column is the copper value and **what numeric range** is acceptable. This governs Task-1 cleaning and everything downstream.

### What the tool looks for
- `--value-col <NAME>`: tell it the exact column name (case-insensitive exact match). When provided, the tool also keeps the common aliases list `["CU_PPM","VALUE"]` as fallbacks.
- `--value-regex <REGEX>`: custom regular expression (optional). If you supply this, it **overrides** the default regex. If you also set `--value-col`, your regex still takes precedence.
- If neither is provided, the **default** detection is:
  - Aliases: `["CU_PPM","VALUE"]`
  - Regex: `^cu_?ppm(_pred)?$` (case-insensitive)

### Range control (outlier filtering)
- `--value-min`: minimum allowed value; default `1e-5`
- `--value-max`: maximum allowed value; default `346000.0`
- Values outside `[min, max]` are dropped during cleaning.

### Priority & behavior
1) If `--value-col` is set → prefer that exact column (case-insensitive).  
2) If `--value-regex` is also set → use your regex for detection (it overrides default).  
3) If neither is set → use defaults (`["CU_PPM","VALUE"]` + `^cu_?ppm(_pred)?$`).

### Good practices
- Make sure your unit is **ppm**. If your raw column is `%` or `mg/kg` with different scale, convert to **ppm** before running.
- Validate by opening the cleaned CSVs (under `reports/task1/cleaned/`) and checking the unified `VALUE` column.

## 1) What you will get
- `reports/task1/cleaned/`
  - `drillhole_original_clean.csv`, `drillhole_dnn_clean.csv`
  - `surface_original_clean.csv`, `surface_dnn_clean.csv`
- `reports/task2/difference/`
  - `{drillhole|surface}_points_{overlap|origonly|dlonly|all}.csv`
- (Optional) `points_all.csv/.parquet` for quick maps.

## 2) What to prepare
- Put **one** ZIP under `data/`.
  - **CSV case**: 4 CSVs (DRILLHOLE/SURFACE × ORIGINAL/DL) inside a single ZIP.
  - **Shapefile case**: 4 shapefile sets (each set = `.shp/.dbf/.shx/...`) packed together into one ZIP.
- Coordinates should be **WGS84 longitude/latitude**.

## 3) Choose one path (non-technical overview)
### Path A — One-click end-to-end (recommended)
Does: staging (if needed) → Task-1 clean → points → Task-2 splits.  
**Tool**: `tools/run_all_test.py` (works for both CSV ZIPs and Shapefile ZIPs).  
Use `--value-col/--value-regex/--value-min/--value-max` here to control the VALUE column and range.

### Path B — Clean only
Does only Task-1 cleaning.  
**Tool**: `tools/run_eda.py` (also supports value options).  
Use when you want cleaned files first; Task-2 later.

### Path C — Recompute Task-2 only
Requires cleaned files already exist.  
**Tool**: `tools/recompute_task2.py`.

### Path D — Build a single `points_all`
**Tools**: `tools/build_points_all.py` or `tools/build_points_all_v2.py` (adds label normalisation).

## 4) Examples for VALUE selection (with `run_all_test`)
**Example 1 — CSV ZIP, value column named `Cu_ppm`, keep 0.01–200000 ppm**
```bash
python tools/run_all_test.py   --raw-zip data/Capstone_CSV.zip   --value-col Cu_ppm   --value-min 0.01   --value-max 200000
```

**Example 2 — Shapefile ZIP, column `GRADE_CU_PPM`, permit 1e-5–300000 ppm**
```bash
python tools/run_all_test.py   --raw-zip data/Capstone_SHP.zip   --value-col GRADE_CU_PPM   --value-min 1e-5   --value-max 300000
```

**Example 3 — When the column name varies, use a regex (two names allowed)**
```bash
python tools/run_all_test.py   --raw-zip data/Capstone_CSV.zip   --value-regex '^(?i:(cu_ppm|cu_ppm_pred))$'   --value-min 0.00001   --value-max 346000
```
*Note*: `(?i:...)` means case-insensitive. If you also pass `--value-col`, the regex still wins.

## 5) Success signs
- Console says **“[EDA] Cleaned CSVs written:”** with the four files listed.
- Rows reported for points and Task-2 outputs.
- No red error messages.

## 6) Common issues
- **No cleaned files found**: run with value options as shown above.
- **Wrong/empty map**: check WGS84 lon/lat; reproject if needed.
- **Value not detected**: verify column name or provide a regex.
