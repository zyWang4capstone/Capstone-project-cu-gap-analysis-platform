# EDA Plan — Task 1

## Objective
Understand dataset structure and quality, explore both basic and comparative patterns between Original and DNN-imputed datasets, and evaluate the nature and significance of differences.  
Prepare the foundation for spatial analysis and model validation.

## Structure
Task 1 is organised into **three parts**, each targeting a distinct analytical objective.

### Part 1 — Dataset Overview & Basic Checks
Focus on loading, validating, and profiling the datasets to establish a clear understanding of structure, completeness, and basic patterns.

#### Steps
1. **Project & Dataset Setup**
   - Unzip and load Original & DNN-imputed CSV files for drillhole and surface datasets.  
   - Verify encoding, delimiters, and column consistency.

2. **Basic Validations**
   - Check data types and record counts.  
   - Verify unique `SAMPLEID` counts and detect duplicates.  
   - Confirm coordinate fields (`LATITUDE`/`LONGITUDE` or `DLAT`/`DLONG`) and coordinate system with client.  
   - Check for NaN, negative, or extreme values in numeric fields.

3. **Key Variables Review**
   - `SAMPLETYPE`, `COLLARID`, depth range and anomalies (e.g., > 2500 m).  
   - Geographic extents and coarse spatial density.

4. **Summary Statistics**
   - Compute descriptive stats for `Cu_ppm`.  
   - Examine distribution patterns for both Original and DNN datasets.

### Part 2 — Comparative Distribution & Spatial Analysis
Assess how DNN imputation changes spatial coverage and value distributions compared to Original datasets.

#### Steps
1. **Spatial Coverage Comparison**
   - Plot Original vs DNN valid points for drillhole and surface datasets.  
   - Summarise changes in record counts and coverage.

2. **Value Distribution Comparison**
   - Compare histograms, log-transformed distributions, and ECDF curves.  
   - Quantify differences via quantiles, KS test, and Cliff’s delta.

3. **Advanced Analysis**
   - KDE overlays in log scale.  
   - Statistical difference testing for both datasets.  
   - Interpret how imputation shifts central tendency, variance, and extremes.

### Part 3 — Record-Level Overlap & Validation Basis
Investigate where Original and DNN datasets contain identical supporting attributes to enable record-by-record comparison.

#### Steps
1. **Overlap Identification**
   - Match records ignoring `SAMPLEID`, using all other shared attributes except `Cu_ppm`.  
   - Calculate proportion of DNN records with a matching Original record.

2. **Validation Strategy**
   - Define “True Value” subset (records with overlap) for direct value comparison.  
   - Define “Regional Estimate” subset (non-overlap) for aggregated or spatial comparison.

3. **Implication**
   - Clarify that record-level overlap is limited (≈5% for drillhole, ≈8% for surface),  
     so most evaluation must rely on spatial or aggregated statistics