# EDA Plan — Task 1

## Objective
Understand dataset structure and quality, perform both basic and comparative exploratory analyses, and prepare multiple cleaned versions for downstream spatial analysis.

## Structure
Task 1 is organised into three Jupyter Notebooks, each targeting a specific analytical goal and producing defined outputs.

### Part 1 — Basic Analysis

#### Steps
1. **Data Loading**
   - Extract `data/raw/csv_datasets.zip` into `data/interim/`
   - Load four CSV files into Pandas DataFrames
   - Verify file encoding and delimiters

2. **Data Inspection**
   - List all fields and check data types
   - Count records per file
   - Identify unique sample IDs and duplicates
   - Verify coordinate fields (`LATITUDE`/`LONGITUDE` or `DLAT`/`DLONG`)
   - Check coordinate reference system (CRS) with client
   - Check numeric fields for NaN, negative values, or extreme values

3. **Descriptive Statistics**
   - Compute mean, median, min, max, standard deviation for `Cu_ppm`
   - Count number of missing or zero `Cu_ppm` values
   - Frequency table of `SAMPLETYPE` and other categorical fields

4. **Basic Visualisation**
   - Scatter plot of coordinates (Original vs DL, side-by-side)
   - Histogram and KDE plot of `Cu_ppm`
   - Depth-binned scatter plot (pseudo-3D by colour coding)
   - Cu concentration heatmap (optional)

#### Outputs
- `summary_statistics.csv` in `reports/`
- Basic charts in `reports/images/part1/`
- Initial `data_dictionary.md` draft

### Part 2 — Comparative Analysis

#### Steps
1. **Data Alignment**
   - Standardise column names between Original and DL datasets
   - Merge datasets by aligning spatial fields (coordinate match) if applicable
   - Ensure units and scales are consistent

2. **Transformation and Distribution Comparison**
   - Apply `log10` transformation to `Cu_ppm`
   - Generate Empirical Cumulative Distribution Function (ECDF) plots
   - Compare Original vs DL ECDF curves

3. **Statistical Testing**
   - Kolmogorov–Smirnov (KS) test for distribution difference
   - Cliff’s delta for effect size

4. **Comparative Visualisation**
   - Side-by-side scatter plots coloured by `Cu_ppm`
   - Difference maps (if applicable)
   - Boxplots or violin plots comparing Original vs DL

#### Outputs
- Comparative analysis plots in `reports/images/part2/`
- Statistical test results in `reports/part2_statistics.csv`
- Markdown summary of differences

### Part 3 — Data Cleaning Scenarios

#### Steps
1. **Define Cleaning Rules**
   - Handling of `SPURIOUS` field (e.g., remove where SPURIOUS > 0)
   - Outlier removal based on Cu threshold (to be confirmed with client)
   - Treatment of missing values (drop or impute)

2. **Generate Alternative Cleaned Datasets**
   - Version A: Remove SPURIOUS records only
   - Version B: Remove SPURIOUS + extreme Cu values
   - Version C: Remove SPURIOUS + extreme Cu + invalid coordinates

3. **Impact Assessment**
   - Record count changes for each cleaning strategy
   - Compare Cu_ppm distributions post-cleaning
   - Spatial distribution changes before/after cleaning

4. **Client Decision Table**
   - Present cleaning rules and resulting dataset summaries
   - Include pros/cons for each scenario

#### Outputs
- Cleaned datasets in `data/processed/` (labelled with version tags)
- Cleaning rule documentation in `reports/part3_cleaning_rules.md`
- Client review checklist

## Client Input Required
- Rule for handling `SPURIOUS`
- Cu_ppm outlier threshold (numeric value)
- Coordinate system confirmation
- Depth unit confirmation

## Final Deliverables
- Processed datasets (`data/processed/`)
- Charts and visualisations (`reports/images/`)
- Three EDA Jupyter Notebooks (`notebooks/`)
- Markdown summaries for each part (`reports/`)
- Client question log (`docs/client_questions.md`)