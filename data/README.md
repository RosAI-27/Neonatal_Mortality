# Data

This directory is reserved for future organization of dataset files.

The current prepared dataset remains at the repository root as `neonatal_mortality_data.csv` because the existing notebook loads it with a root-relative path. Keeping that path unchanged preserves the reproducibility of the current analysis.

## Dataset

- **Source:** EDS/DHS Cameroon 2018
- **Prepared file:** `neonatal_mortality_data.csv`
- **Separator:** `;`
- **Target variable:** `neonatal_mort`
- **Observations:** 33,988
- **Variables:** 15 in the prepared analytical dataset

The Streamlit application also supports uploading a CSV through its sidebar.