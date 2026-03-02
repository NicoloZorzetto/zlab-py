# zlab-py
Python port of the zlab library


Its core module, **`zform`**, automatically identifies the best parametric transformation that linearizes the relationship between variables.  
It explores multiple functional forms (linear, log, power, logistic, etc.), fits them using `scipy.curve_fit`, and evaluates their performance via metrics such as R², RMSE, AIC, and BIC.

---

# Installation
```bash
git clone https://github.com/<your-username>/zlab-py.git
cd zlab-py
pip install -r requirements.txt
```

Then in Python:
```python
from zlab import zform
```

---

# Dependencies
Core: numpy, pandas, scipy  
Optional (for tests): seaborn

---

# zform
## Features
- Automatically tests multiple transformation families
- Dynamic parameter initialization for robust optimization
- Supports both fixed and discovery (fitted) log bases
- Parallelized pairwise fitting with `ProcessPoolExecutor`
- Grouped analysis (e.g., by species or category)
- Exports best-fit results and coefficients to CSV  
- Optional application of best transformations to the original DataFrame

---

## Example usage
```python
from zlab import zform
import seaborn as sbn

df = sbn.load_dataset("penguins").dropna()
df_out, zforms = zform(
    df,
    group_col="species",
	min_obs = 30,
	apply = True,
	naming = "standard",
    export_csv="./forms.csv",
    n_jobs=-1
)

print(zforms.summary())
```

---

# Dataset Credits

This package includes copies of the following datasets, under `src/zlab/datasets/`, used for testing and examples. The datasets are redistributed under their original terms; no modifications were made.

## Iris Dataset

Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems*. *Annals of Eugenics*, 7(2), 179–188.

A widely used copy is available via the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/53/iris

The data is in the public domain; no license restrictions apply.

## Palmer Penguins Dataset

Gorman KB, Williams TD, Fraser WR (2014). *Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis)*. *PLOS ONE*, 9(3), e90081.
https://doi.org/10.1371/journal.pone.0090081

Commonly cited via the `palmerpenguins` R package:
https://allisonhorst.github.io/palmerpenguins/

Released under CC0 1.0 (Public Domain):
https://creativecommons.org/publicdomain/zero/1.0/

---

# License
GPL v3 — see LICENSE for details.

---

# Author
**Nicolò Zorzetto**