# 🚗 Bayesian Causal Discovery on the Auto MPG Dataset

This project applies Bayesian causal inference techniques to the [Auto MPG dataset](https://github.com/mwaskom/seaborn-data/blob/master/mpg.csv), combining constraint-based structure discovery using the PC algorithm and probabilistic reasoning with Bayesian networks. We use both observational queries and simulated interventions to explore relationships between engine performance metrics and fuel efficiency.

---

## 🧠 Project Summary

### **Key goals:**
- Discover causal relationships between automotive features like `mpg`, `horsepower`, and `weight`
- Model the system using a discrete Bayesian network
- Answer queries such as:
  - *What’s the probability distribution of `mpg` given `weight` and `acceleration`?*
  - *What is the effect of setting `displacement = low` on `cylinders`?*

### We leverage:
- **PC algorithm** from [`causal-learn`](https://github.com/py-why/causal-learn) for structure discovery
- **`pgmpy`** for Bayesian parameter estimation, inference, and interventional reasoning

---

## 📁 Files

| File | Description |
|------|-------------|
| `bayesian_causality.ipynb` | Main notebook containing code, graphs, queries, and interventions |
| `requirements.txt` | List of required Python packages |
| `LICENSE` | MIT License |
| `.gitignore` | Ignore Jupyter checkpoints and system files |

---

## 📊 Dataset

The [MPG dataset](https://github.com/mwaskom/seaborn-data/blob/master/mpg.csv) includes attributes for 1970s–1980s US automobiles. We focus on the following numeric variables:

- `mpg` (miles per gallon)
- `cylinders`
- `displacement`
- `horsepower`
- `weight`
- `acceleration`

Missing values are dropped. Continuous values are discretized into categories (low, medium, high) using quantile binning.

---

## 🔬 Methodology

### 1. **Causal Discovery**

We apply the **PC algorithm** using the Fisher Z conditional independence test to discover the undirected structure from continuous numeric data.

### 2. **Bayesian Network Construction**

- Selected edges are manually defined based on interpretability and the PC graph.
- Variables are discretized.
- Parameters (CPDs) are estimated using Bayesian smoothing (`BDeu` prior).

### 3. **Inference**

We use **Variable Elimination** to answer conditional queries:

```python
P(mpg | weight='high', acceleration='low')
```

### 4. **Interventions (do() operator)**
Simulated interventions are done by:

Removing parents of the target variable

Fixing it to a specific value

Performing inference again on the modified network

Example:
```
P(cylinders | do(displacement='low'))
```

## 📈 Sample Output
### ➤ Conditional Query

```
# Query: What is the distribution of mpg given weight=high and acceleration=low?
query = infer.query(variables=['mpg'], evidence={'weight': 'high', 'acceleration': 'low'})
print(query)
```

| mpg           | φ(mpg)   |
|---------------|----------|
| mpg(high)     | 0.0023   |
| mpg(low)      | 0.9205   |
| mpg(medium)   | 0.0771   |


### ➤ Interventional Query
```
# Intervention: What is the distribution of cylinders after setting displacement=low?
result = do_intervention(model, intervention_var='displacement', intervention_value='low', query_var='cylinders')
print(result)
```

| cylinders         | φ(cylinders) |
|-------------------|--------------|
| cylinders(high)   | 0.2636       |
| cylinders(low)    | 0.5155       |
| cylinders(medium) | 0.2208       |


## 🛠️ How to Run
### 1. Clone the repository
```
git clone https://github.com/Dimejitech/bayesian-causality-mpg.git
cd bayesian-causality-mpg
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Launch the notebook
```
jupyter notebook bayesian_causality_mpg.ipynb
```

## 🧱 Requirements
- Python 3.8+

- pandas

- numpy

- matplotlib

- pgmpy

- causallearn

- graphviz (optional, for rendering graphs)

## 📚 References
Pearl, J. (2009). *Causality: Models, Reasoning and Inference.*

- [causal-learn GitHub repo](https://github.com/py-why/causal-learn)
- [pgmpy documentation](https://pgmpy.org/)
- [Auto MPG dataset (seaborn)](https://github.com/mwaskom/seaborn-data/blob/master/mpg.csv)

## 👤 Author
**Oladimeji Abaniwonnda**
Computer Science @ Pan-Atlantic University
[LinkedIn](https://www.linkedin.com/in/oladimeji-abaniwonnda) | [GitHub](https://github.com/Dimejitech)

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
