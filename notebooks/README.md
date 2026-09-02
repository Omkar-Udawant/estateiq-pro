# EstateIQ Pro — Exploratory Data Analysis

Run cells here to explore King County data. **Production code lives in `src/`** — promote stable transforms from this notebook into `src/preprocessing.py`.

```python
# Quick start
import sys
sys.path.insert(0, "..")
from src.data_loader import load_processed_data
df = load_processed_data()
df.describe()
```
