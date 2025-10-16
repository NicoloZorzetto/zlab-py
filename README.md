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
df_out, results = zform(
    df,
    group_col="species",
    return_results=True,
	min_obs = 30,
	apply = True,
	naming = "standard",
    export_csv="./forms.csv",
    n_jobs=-1
)

print(results.query('y == "bill_length_mm" and x == "body_mass_g"'))
```

---

## Quick test
Run the built-in test script:

```bash
python3 tests/test_zform.py
```

It automatically loads the penguins dataset, runs zform, and prints the best-fitting models.

---

# License
GPL v3 — see LICENSE for details.

---

# Author
**Nicolò Zorzetto**