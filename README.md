# Counter-Strike-round-winner-prediciton
Machine learning classification project that predicts CS:GO round winners using gameplay features, comparing Logistic Regression, Decision Tree, Random Forest, and KNN models.

# 🎮 CS:GO Round Winner Prediction Using Machine Learning

A machine learning classification project that predicts the winner of a **Counter-Strike: Global Offensive (CS:GO)** round using in-game state information such as team scores, health, armor, money, weapons, grenades, map, remaining time, bomb status, and player information.

The project performs exploratory data analysis, data preprocessing, categorical encoding, feature/target separation, train-test splitting, feature scaling, Linear Discriminant Analysis (LDA), and classification using multiple machine learning algorithms.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Project Objective](#-project-objective)
* [Dataset](#-dataset)
* [Dataset Structure](#-dataset-structure)
* [Features](#-features)
* [Target Variable](#-target-variable)
* [Technologies Used](#-technologies-used)
* [Project Workflow](#-project-workflow)
* [1. Import Libraries](#1-import-libraries)
* [2. Load the Dataset](#2-load-the-dataset)
* [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
* [4. Dataset Information](#4-dataset-information)
* [5. Missing Value Analysis](#5-missing-value-analysis)
* [6. Map Analysis](#6-map-analysis)
* [7. Categorical Encoding](#7-categorical-encoding)
* [8. Feature and Target Separation](#8-feature-and-target-separation)
* [9. Train-Test Split](#9-train-test-split)
* [10. Feature Scaling](#10-feature-scaling)
* [11. Linear Discriminant Analysis](#11-linear-discriminant-analysis)
* [12. LDA Coefficient Analysis](#12-lda-coefficient-analysis)
* [13. Machine Learning Models](#13-machine-learning-models)
* [14. Model Performance](#14-model-performance)
* [15. Final Result](#15-final-result)
* [Installation](#-installation)
* [How to Run the Project](#-how-to-run-the-project)
* [Running on Google Colab](#-running-on-google-colab)
* [Running Locally](#-running-locally)
* [Project Structure](#-project-structure)
* [Complete Reproduction Workflow](#-complete-reproduction-workflow)
* [Important Notes](#-important-notes)
* [Future Improvements](#-future-improvements)
* [Conclusion](#-conclusion)
* [Author](#-author)

---

# 🎯 Project Overview

Counter-Strike: Global Offensive (CS:GO) is a team-based tactical first-person shooter in which two teams compete:

* **Counter-Terrorists (CT)**
* **Terrorists (T)**

The objective of this project is to use machine learning to predict which team will win a round based on the current state of the game.

The dataset contains information about the game state at different points during rounds, including:

* Remaining round time
* Current team scores
* Team health
* Team armor
* Team money
* Helmets
* Defuse kits
* Number of alive players
* Weapons available to each team
* Grenades available to each team
* Map
* Whether the bomb has been planted
* Actual round winner

The project treats **round winner prediction as a binary classification problem**.

---

# 🎯 Project Objective

The main objective is:

> **Predict whether the Counter-Terrorist team (CT) or Terrorist team (T) will win a CS:GO round using the available game-state features.**

The notebook implements and compares the following machine learning algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)

The project also applies:

* Exploratory Data Analysis
* Label Encoding
* Train/Test Split
* Standardization
* Linear Discriminant Analysis
* LDA coefficient analysis

---

# 📊 Dataset

The project uses the CSV file:

```text
DataCGGO.csv
```

The uploaded dataset used in this project contains:

* **122,410 rows**
* **97 columns**

The notebook loads the dataset using:

```python
import pandas as pd

df = pd.read_csv('/content/DataCGGO.csv')
```

When running locally, change the path to the location of your CSV file.

For example:

```python
df = pd.read_csv('DataCGGO.csv')
```

The dataset contains both numerical and categorical/binary information.

The original dataframe contains:

```text
122,410 rows × 97 columns
```

The notebook's `df.describe()` output contains statistics for the numerical columns, while `df.info()` is used to inspect data types and non-null counts.

---

# 🧩 Dataset Structure

The dataset contains **97 columns**.

These can broadly be grouped into the following categories:

### Game State

* `time_left`
* `ct_score`
* `t_score`
* `map`
* `bomb_planted`

### Team Health

* `ct_health`
* `t_health`

### Team Armor

* `ct_armor`
* `t_armor`

### Team Economy

* `ct_money`
* `t_money`

### Helmets

* `ct_helmets`
* `t_helmets`

### Defuse Kits

* `ct_defuse_kits`

### Players Alive

* `ct_players_alive`
* `t_players_alive`

### Weapons

The dataset contains separate weapon-related features for CT and T teams.

### Grenades

The dataset contains separate grenade-related features for CT and T teams.

### Target

* `round_winner`

---

# 📝 Complete Feature List

The 97 columns in the dataset are:

```text
time_left
ct_score
t_score
map
bomb_planted
ct_health
t_health
ct_armor
t_armor
ct_money
t_money
ct_helmets
t_helmets
ct_defuse_kits
ct_players_alive
t_players_alive
ct_weapon_ak47
t_weapon_ak47
ct_weapon_aug
t_weapon_aug
ct_weapon_awp
t_weapon_awp
ct_weapon_bizon
t_weapon_bizon
ct_weapon_cz75auto
t_weapon_cz75auto
ct_weapon_elite
t_weapon_elite
ct_weapon_famas
t_weapon_famas
ct_weapon_g3sg1
t_weapon_g3sg1
ct_weapon_galilar
t_weapon_galilar
ct_weapon_glock
t_weapon_glock
ct_weapon_m249
t_weapon_m249
ct_weapon_m4a1s
t_weapon_m4a1s
ct_weapon_m4a4
t_weapon_m4a4
ct_weapon_mac10
t_weapon_mac10
ct_weapon_mag7
t_weapon_mag7
ct_weapon_mp5sd
t_weapon_mp5sd
ct_weapon_mp7
t_weapon_mp7
ct_weapon_mp9
t_weapon_mp9
ct_weapon_negev
t_weapon_negev
ct_weapon_nova
t_weapon_nova
ct_weapon_p90
t_weapon_p90
ct_weapon_r8revolver
t_weapon_r8revolver
ct_weapon_sawedoff
t_weapon_sawedoff
ct_weapon_scar20
t_weapon_scar20
ct_weapon_sg553
t_weapon_sg553
ct_weapon_ssg08
t_weapon_ssg08
ct_weapon_ump45
t_weapon_ump45
ct_weapon_xm1014
t_weapon_xm1014
ct_weapon_deagle
t_weapon_deagle
ct_weapon_fiveseven
t_weapon_fiveseven
ct_weapon_usps
t_weapon_usps
ct_weapon_p250
t_weapon_p250
ct_weapon_p2000
t_weapon_p2000
ct_weapon_tec9
t_weapon_tec9
ct_grenade_hegrenade
t_grenade_hegrenade
ct_grenade_flashbang
t_grenade_flashbang
ct_grenade_smokegrenade
t_grenade_smokegrenade
ct_grenade_incendiarygrenade
t_grenade_incendiarygrenade
ct_grenade_molotovgrenade
t_grenade_molotovgrenade
ct_grenade_decoygrenade
t_grenade_decoygrenade
round_winner
```

The notebook confirms that after removing the target column, there are **96 input features**.

---

# 🎯 Target Variable

The target variable is:

```text
round_winner
```

It represents the team that won the round.

The original dataset contains two classes:

```text
CT
T
```

The target is converted into numerical values using `LabelEncoder`.

The encoding performed in the notebook results in:

```text
CT → 0
T  → 1
```

The dataset contains:

```text
T  = 62,406
CT = 60,004
```

This makes the classification problem a binary classification task.

---

# 🛠 Technologies Used

The project uses Python and the following libraries:

### Python

Python is used for data processing, analysis, visualization, and machine learning.

### Pandas

Used for:

* Loading the CSV dataset
* DataFrame manipulation
* Descriptive statistics
* Missing-value analysis
* Column inspection

```python
import pandas as pd
```

### NumPy

Used for numerical operations.

```python
import numpy as np
```

### Seaborn

Imported for statistical visualization.

```python
import seaborn as sns
```

### Matplotlib

Imported for visualization.

```python
import matplotlib.pyplot as plt
```

### Scikit-learn

Used for:

* Label encoding
* Train/test splitting
* Feature scaling
* LDA
* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* Accuracy evaluation

---

# 🔄 Project Workflow

The complete workflow used in this project is:

```text
Dataset
   ↓
Load CSV
   ↓
Inspect Dataset
   ↓
Descriptive Statistics
   ↓
Check Columns
   ↓
Check Data Types
   ↓
Check Missing Values
   ↓
Explore Maps
   ↓
Encode Categorical Variables
   ↓
Separate Features and Target
   ↓
Train-Test Split
   ↓
Standardization
   ↓
Linear Discriminant Analysis
   ↓
LDA Coefficient Analysis
   ↓
Train Classification Models
   ↓
Evaluate Accuracy
   ↓
Compare Models
```

---

# 1. Import Libraries

The notebook begins by importing the primary data analysis and visualization libraries:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

These libraries provide the foundation for the analysis.

---

# 2. Load the Dataset

The notebook loads the dataset into a Pandas DataFrame:

```python
df = pd.read_csv('/content/DataCGGO.csv')
```

If you are using Google Colab, the dataset can be uploaded and the path adjusted accordingly.

If you are running the project locally:

```python
df = pd.read_csv('DataCGGO.csv')
```

---

# 3. Exploratory Data Analysis

The first inspection is performed by displaying the DataFrame:

```python
df
```

The resulting dataset contains:

```text
122,410 rows
97 columns
```

The first rows show game-state information such as:

```text
time_left
ct_score
t_score
map
bomb_planted
ct_health
t_health
ct_armor
t_armor
ct_money
...
```

The final column is:

```text
round_winner
```

---

# 4. Dataset Information

The project uses:

```python
df.describe()
```

to obtain statistical information about numerical columns.

Examples from the analysis include:

| Feature     |       Mean | Minimum | Maximum |
| ----------- | ---------: | ------: | ------: |
| `time_left` |    97.8869 |    0.01 |     175 |
| `ct_score`  |     6.7092 |       0 |      32 |
| `t_score`   |     6.7804 |       0 |      33 |
| `ct_health` |   412.1066 |       0 |     500 |
| `t_health`  |   402.7145 |       0 |     600 |
| `ct_armor`  |   314.1421 |       0 |     500 |
| `t_armor`   |   298.4447 |       0 |     500 |
| `ct_money`  |  9789.0238 |       0 |   80000 |
| `t_money`   | 11241.0367 |       0 |   80000 |

The notebook also checks the columns:

```python
df.columns
```

and dataset information:

```python
df.info()
```

The dataset contains:

```text
122,410 entries
97 columns
```

---

# 5. Missing Value Analysis

The notebook checks missing values using:

```python
df.isnull().sum()
```

and then calculates the total number of missing values:

```python
df.isnull().sum().sum()
```

The result is:

```text
0
```

Therefore, the dataset contains **no missing values** according to the analysis performed in the notebook.

This means no missing-value imputation was required.

---

# 6. Map Analysis

The project investigates the maps available in the dataset.

The notebook uses:

```python
df['map'].unique()
```

The dataset contains the following maps:

```text
de_dust2
de_mirage
de_nuke
de_inferno
de_overpass
de_vertigo
de_train
de_cache
```

The notebook then calculates the number of records for each map:

```python
df['map'].value_counts()
```

The recorded counts are:

| Map           | Records |
| ------------- | ------: |
| `de_inferno`  |  23,811 |
| `de_dust2`    |  22,144 |
| `de_nuke`     |  19,025 |
| `de_mirage`   |  18,576 |
| `de_overpass` |  14,081 |
| `de_train`    |  13,491 |
| `de_vertigo`  |  11,137 |
| `de_cache`    |     145 |

`de_inferno` has the highest number of observations in this dataset, while `de_cache` has the fewest.

---

# 7. Categorical Encoding

Machine learning algorithms require numerical representations of categorical values.

The project imports:

```python
from sklearn.preprocessing import LabelEncoder
```

An encoder object is created:

```python
encoder = LabelEncoder()
```

The notebook then encodes three columns:

```python
df['map'] = encoder.fit_transform(df['map'])

df['bomb_planted'] = encoder.fit_transform(df['bomb_planted'])

df['round_winner'] = encoder.fit_transform(df['round_winner'])
```

The categorical variables are therefore transformed into numerical values.

For the map column, alphabetical LabelEncoder ordering produces:

```text
de_cache    → 0
de_dust2    → 1
de_inferno  → 2
de_mirage   → 3
de_nuke     → 4
de_overpass → 5
de_train    → 6
de_vertigo  → 7
```

For `bomb_planted`:

```text
False → 0
True  → 1
```

For `round_winner`:

```text
CT → 0
T  → 1
```

---

# 8. Feature and Target Separation

The project separates the independent variables from the target variable.

The target is removed from the feature dataset:

```python
x = df.drop(columns=['round_winner'])
```

The target is assigned to:

```python
y = df[['round_winner']]
```

Therefore:

```text
X = 96 input features
Y = round_winner
```

The project contains:

```text
96 predictor variables
1 target variable
```

---

# 9. Train-Test Split

The project uses Scikit-learn's `train_test_split`:

```python
from sklearn.model_selection import train_test_split
```

The dataset is split using:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=45
)
```

The split uses:

```text
Training data = 80%
Testing data  = 20%
Random state  = 45
```

With 122,410 total observations:

```text
Training samples = 97,928
Testing samples  = 24,482
```

The training and testing datasets are therefore:

```text
X_train → 97,928 rows × 96 features
X_test  → 24,482 rows × 96 features

y_train → 97,928 rows
y_test  → 24,482 rows
```

The use of:

```python
random_state=45
```

makes the split reproducible.

---

# 10. Feature Scaling

The notebook uses `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler
```

A scaler object is created:

```python
sc = StandardScaler()
```

The training data is fitted and transformed:

```python
x_train = sc.fit_transform(x_train)
```

The test data is transformed using the same scaler:

```python
x_test = sc.transform(x_test)
```

This is important because the scaler learns the transformation parameters from the training data and applies the same transformation to the test data.

The resulting features are standardized.

---

# 11. Linear Discriminant Analysis

The project also applies Linear Discriminant Analysis.

The required class is imported:

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

The model is initialized:

```python
ldaObj = LinearDiscriminantAnalysis()
```

The model is fitted:

```python
ldaObj.fit(x_train, y_train)
```

The notebook then transforms the test data:

```python
ldaObj.transform(x_test)
```

The resulting output is a one-dimensional discriminant representation because this is a binary classification problem.

The notebook therefore uses LDA to examine the separation of the two round-winner classes.

---

# 12. LDA Coefficient Analysis

The project calculates an LDA coefficient-based score:

```python
ldaCoef = np.exp(np.abs(ldaObj.coef_))
```

The notebook comment explains that the purpose is to calculate an LDA coefficient score that can be used to assess the importance associated with the features.

The array is then flattened:

```python
ldaCoef = ldaCoef.flatten()
```

The number of features is calculated using:

```python
num_feat = x.shape[1]
```

The result is:

```text
96
```

Therefore, the model is working with **96 input features**.

---

# 13. Machine Learning Models

The project evaluates four classification algorithms.

---

## 13.1 Logistic Regression

The first model is Logistic Regression.

The notebook imports:

```python
from sklearn.linear_model import LogisticRegression
```

The model is created:

```python
lr = LogisticRegression()
```

It is trained using:

```python
lr.fit(x_train, y_train)
```

Predictions are generated:

```python
y_pred = lr.predict(x_test)
```

The accuracy is calculated using:

```python
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(accuracy_score(y_test, y_pred))
```

### Logistic Regression Accuracy

```text
0.7481414917081938
```

Approximately:

```text
74.81%
```

---

# 13.2 Decision Tree Classifier

The second model is a Decision Tree.

The notebook imports:

```python
from sklearn.tree import DecisionTreeClassifier
```

The model is initialized:

```python
dt = DecisionTreeClassifier()
```

It is trained:

```python
dt.fit(x_train, y_train)
```

Predictions are generated:

```python
y_pred = dt.predict(x_test)
```

The accuracy is then calculated.

### Decision Tree Accuracy

```text
0.8237072134629524
```

Approximately:

```text
82.37%
```

---

# 13.3 Random Forest Classifier

The third model is Random Forest.

The notebook imports:

```python
from sklearn.ensemble import RandomForestClassifier
```

The model is initialized:

```python
rf = RandomForestClassifier()
```

It is trained:

```python
rf.fit(x_train, y_train)
```

Predictions are generated:

```python
y_pred = rf.predict(x_test)
```

The accuracy is calculated using:

```python
print(accuracy_score(y_test, y_pred))
```

### Random Forest Accuracy

```text
0.8765623723551997
```

Approximately:

```text
87.66%
```

---

# 13.4 K-Nearest Neighbors

The fourth model is K-Nearest Neighbors.

The notebook imports:

```python
from sklearn.neighbors import KNeighborsClassifier
```

The model is initialized:

```python
kn = KNeighborsClassifier()
```

The default number of neighbors is:

```text
5
```

This is confirmed in the notebook using:

```python
kn.n_neighbors
```

The model is trained:

```python
kn.fit(x_train, y_train)
```

Predictions are generated:

```python
y_pred = kn.predict(x_test)
```

The accuracy is calculated.

### KNN Accuracy

```text
0.8333469487786945
```

Approximately:

```text
83.33%
```

---

# 📊 14. Model Performance

The recorded model performance from the notebook is:

| Model               | Accuracy | Accuracy (%) |
| ------------------- | -------: | -----------: |
| Logistic Regression | 0.748141 |       74.81% |
| Decision Tree       | 0.823707 |       82.37% |
| K-Nearest Neighbors | 0.833347 |       83.33% |
| Random Forest       | 0.876562 |   **87.66%** |

### Model Ranking

Based on the recorded test accuracy:

```text
1. Random Forest          → 87.66%
2. K-Nearest Neighbors    → 83.33%
3. Decision Tree          → 82.37%
4. Logistic Regression    → 74.81%
```

---

# 🏆 15. Final Result

Among the four models tested in the notebook, **Random Forest Classifier achieved the highest recorded test accuracy**.

Its accuracy was:

```text
87.65623723551997%
```

or approximately:

```text
87.66%
```

Therefore, based on the experiments performed in this notebook, Random Forest produced the best classification accuracy among the tested models.

---

# 💻 Installation

To run this project on your own system, install Python first.

Python 3.x is recommended.

Then install the required packages.

```bash
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
```

Alternatively:

```bash
pip install -r requirements.txt
```

You can create a `requirements.txt` file containing:

```text
pandas
numpy
seaborn
matplotlib
scikit-learn
jupyter
```

---

# 📁 Project Structure

A recommended GitHub repository structure is:

```text
CSGO-Round-Winner-Prediction/
│
├── Couter Strike(csgo)_project(1).ipynb
│
├── DataCGGO(1).csv
│
├── README.md
│
└── requirements.txt
```

For a cleaner GitHub repository, you can rename the files to:

```text
CSGO-Round-Winner-Prediction/
│
├── csgo_round_winner_prediction.ipynb
├── DataCGGO.csv
├── README.md
└── requirements.txt
```

Renaming the notebook and dataset is optional.

---

# ▶️ How to Run the Project

There are two recommended ways to run this project:

1. Google Colab
2. Local Jupyter Notebook

---

# ☁️ Running on Google Colab

The original notebook was developed in a Google Colab environment.

To run it:

### Step 1 — Open Google Colab

Open Google Colab in your browser.

### Step 2 — Upload the notebook

Upload:

```text
Couter Strike(csgo)_project(1).ipynb
```

### Step 3 — Upload the dataset

Upload:

```text
DataCGGO(1).csv
```

### Step 4 — Check the dataset path

The original notebook uses:

```python
df = pd.read_csv('/content/DataCGGO.csv')
```

If your uploaded file has a different name, change the path accordingly.

For example:

```python
df = pd.read_csv('/content/DataCGGO(1).csv')
```

### Step 5 — Run all cells

Run:

```text
Runtime → Run all
```

The notebook will execute:

```text
Data Loading
↓
EDA
↓
Missing Value Check
↓
Map Analysis
↓
Encoding
↓
Feature/Target Separation
↓
Train/Test Split
↓
Scaling
↓
LDA
↓
Model Training
↓
Accuracy Evaluation
```

---

# 🖥️ Running Locally

## Step 1 — Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Move into the project directory:

```bash
cd CSGO-Round-Winner-Prediction
```

---

## Step 2 — Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
```

---

## Step 4 — Start Jupyter Notebook

Run:

```bash
jupyter notebook
```

A browser window will open.

Open:

```text
Couter Strike(csgo)_project(1).ipynb
```

---

# 📂 Dataset Path When Running Locally

The original notebook contains:

```python
df = pd.read_csv('/content/DataCGGO.csv')
```

The `/content/` path is specific to Google Colab.

When running locally, change it to:

```python
df = pd.read_csv('DataCGGO.csv')
```

If the dataset is stored in a separate folder:

```python
df = pd.read_csv('data/DataCGGO.csv')
```

For Windows, an absolute path can also be used:

```python
df = pd.read_csv(r'C:\path\to\your\project\DataCGGO.csv')
```

Using a relative path is recommended because it makes the project easier to share.

---

# 🔁 Complete Reproduction Workflow

The following is the complete sequence used by the project.

## Import libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

## Load data

```python
df = pd.read_csv('DataCGGO.csv')
```

## Inspect data

```python
df
```

## Descriptive statistics

```python
df.describe()
```

## Inspect columns

```python
df.columns
```

## Inspect data types and structure

```python
df.info()
```

## Check missing values

```python
df.isnull().sum()
```

## Check total missing values

```python
df.isnull().sum().sum()
```

## Check unique maps

```python
df['map'].unique()
```

## Count map observations

```python
df['map'].value_counts()
```

## Import LabelEncoder

```python
from sklearn.preprocessing import LabelEncoder
```

## Create encoder

```python
encoder = LabelEncoder()
```

## Encode categorical columns

```python
df['map'] = encoder.fit_transform(df['map'])

df['bomb_planted'] = encoder.fit_transform(df['bomb_planted'])

df['round_winner'] = encoder.fit_transform(df['round_winner'])
```

## Separate features and target

```python
x = df.drop(columns=['round_winner'])

y = df[['round_winner']]
```

## Split data

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=45
)
```

## Standardize features

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
```

## LDA

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ldaObj = LinearDiscriminantAnalysis()

ldaObj.fit(x_train, y_train)

ldaObj.transform(x_test)
```

## LDA coefficient analysis

```python
ldaCoef = np.exp(np.abs(ldaObj.coef_))

ldaCoef = ldaCoef.flatten()
```

## Number of features

```python
num_feat = x.shape[1]

print(num_feat)
```

Expected result:

```text
96
```

---

# 🤖 Train the Models

## Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
```

Expected recorded accuracy:

```text
0.7481414917081938
```

---

## Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

print(accuracy_score(y_test, y_pred))
```

Expected recorded accuracy:

```text
0.8237072134629524
```

---

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

print(accuracy_score(y_test, y_pred))
```

Expected recorded accuracy:

```text
0.8765623723551997
```

---

## KNN

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

kn.fit(x_train, y_train)

y_pred = kn.predict(x_test)

print(accuracy_score(y_test, y_pred))
```

Expected recorded accuracy:

```text
0.8333469487786945
```

Check the default number of neighbors:

```python
kn.n_neighbors
```

Expected:

```text
5
```

---

# ⚠️ Important Notes

## 1. The notebook uses a Google Colab path

The original notebook contains:

```python
/content/DataCGGO.csv
```

This path will not normally exist on a local Windows/Linux/macOS system.

Change it to:

```python
DataCGGO.csv
```

or the appropriate path on your computer.

---

## 2. LabelEncoder is fitted separately

The notebook uses:

```python
encoder.fit_transform()
```

separately on:

```text
map
bomb_planted
round_winner
```

This is important because the encoder is reused but refitted for each column.

If you modify the preprocessing pipeline, preserve the mapping between the encoded values and their original categories.

---

## 3. Target leakage considerations

The dataset contains multiple game-state variables that can be highly informative about the eventual winner.

This project uses the features exactly as present in the notebook and predicts:

```text
round_winner
```

When extending the project, carefully consider whether every feature would genuinely be available at the exact moment a real-world prediction is supposed to be made.

---

## 4. Reproducibility

The train/test split uses:

```python
random_state=45
```

This helps reproduce the same train/test partition.

However, some models such as Random Forest and other algorithms may have additional randomness unless their `random_state` is explicitly set.

The original notebook does not specify a `random_state` for the individual classifiers.

Therefore, small differences in model results may occur when rerunning the notebook with different library versions or environments.

---

## 5. DataConversionWarning

The notebook produces Scikit-learn warnings such as:

```text
DataConversionWarning:
A column-vector y was passed when a 1d array was expected.
```

This occurs because the target is created as:

```python
y = df[['round_winner']]
```

which produces a DataFrame rather than a one-dimensional Series.

A cleaner implementation would be:

```python
y = df['round_winner']
```

This is an implementation improvement and does not change the project's conceptual objective.

---

# 🚀 Future Improvements

The current project provides a solid classification workflow, but it can be extended further.

Possible improvements include:

### 1. Hyperparameter Tuning

Optimize model parameters using:

```text
GridSearchCV
RandomizedSearchCV
```

This can be applied to:

* Random Forest
* Decision Tree
* KNN
* Logistic Regression

---

### 2. Cross-Validation

Instead of relying on one train/test split, use:

```text
K-Fold Cross Validation
Stratified K-Fold Cross Validation
```

to obtain more robust performance estimates.

---

### 3. Additional Evaluation Metrics

The current notebook primarily evaluates models using accuracy.

Additional metrics could include:

```text
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
Classification Report
```

For example:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

---

### 4. Feature Importance

The Random Forest model can be used to investigate feature importance:

```python
feature_importance = pd.Series(
    rf.feature_importances_,
    index=x.columns
).sort_values(ascending=False)

print(feature_importance)
```

This can help identify which game-state variables contribute most to prediction.

---

### 5. Hyperparameter Optimization for Random Forest

Possible parameters to tune include:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

---

### 6. Model Visualization

Additional visualizations can be added for:

* Class distribution
* Map distribution
* Feature distributions
* Correlation matrix
* Confusion matrices
* Feature importance
* ROC curves
* Precision-recall curves

---

### 7. Prediction Interface

The project could eventually be converted into an interactive application using:

```text
Streamlit
Flask
FastAPI
```

A user could enter the current game state and receive a prediction such as:

```text
Predicted Winner: CT
```

or:

```text
Predicted Winner: T
```

---

# 🧪 Example Prediction Workflow

Once a trained model is available, a new game-state observation can be processed using the same preprocessing pipeline.

The general workflow is:

```text
New Game State
      ↓
Categorical Encoding
      ↓
Feature Ordering
      ↓
StandardScaler
      ↓
Random Forest
      ↓
Predicted Round Winner
```

The most important requirement is that the new observation must contain the same 96 input features used during model training.

---

# 📌 Model Comparison Summary

The project demonstrates that different machine learning algorithms perform differently on the same CS:GO game-state classification problem.

### Logistic Regression

```text
Accuracy: 74.81%
```

Provides the lowest recorded accuracy among the tested models.

### Decision Tree

```text
Accuracy: 82.37%
```

Performs substantially better than Logistic Regression.

### KNN

```text
Accuracy: 83.33%
```

Performs slightly better than the Decision Tree in the recorded experiment.

### Random Forest

```text
Accuracy: 87.66%
```

Provides the highest recorded accuracy among the models tested in the notebook.

---

# 🏅 Best Performing Model

Based on the recorded experiment:

```text
Random Forest Classifier
```

is the best-performing model among the four tested algorithms.

Recorded test accuracy:

```text
87.65623723551997%
```

Rounded:

```text
87.66%
```

---

# 📚 Skills Demonstrated

This project demonstrates practical experience with:

* Python
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Exploratory Data Analysis
* Data Cleaning
* Missing Value Analysis
* Categorical Encoding
* Feature Engineering / Feature Preparation
* Train-Test Splitting
* Feature Scaling
* StandardScaler
* Linear Discriminant Analysis
* Logistic Regression
* Decision Trees
* Random Forest
* K-Nearest Neighbors
* Binary Classification
* Model Evaluation
* Accuracy Comparison
* Machine Learning Workflow

---

# 💼 Data Science Concepts Demonstrated

From a Data Science perspective, the project follows a standard machine learning pipeline:

```text
Business/Problem Understanding
        ↓
Data Collection
        ↓
Data Loading
        ↓
Exploratory Data Analysis
        ↓
Data Quality Checking
        ↓
Categorical Encoding
        ↓
Feature/Target Selection
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
Model Development
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Best Model Selection
```

---

# 📈 Key Findings

Based strictly on the analysis performed in the notebook:

1. The dataset contains **122,410 observations**.
2. The dataset contains **97 columns**.
3. There are **96 predictor features** after removing `round_winner`.
4. There are no missing values according to the missing-value checks.
5. The dataset contains eight maps.
6. `de_inferno` has the largest number of observations.
7. `de_cache` has the smallest number of observations.
8. The target variable is `round_winner`.
9. The target has two classes: CT and T.
10. The project uses an 80/20 train-test split.
11. The split uses `random_state=45`.
12. StandardScaler is used before model training.
13. LDA is used for discriminant analysis.
14. Four classification models are evaluated.
15. Random Forest achieves the highest recorded test accuracy.
16. The recorded Random Forest accuracy is approximately **87.66%**.

---

# 🔍 Limitations

The current project has several limitations that should be considered when interpreting the results.

### Single Train/Test Split

The evaluation uses one 80/20 train-test split.

Cross-validation would provide a more robust estimate of generalization performance.

### Limited Hyperparameter Tuning

The classifiers are initialized with their default parameters.

No systematic hyperparameter optimization is performed in the original notebook.

### Accuracy as Primary Metric

The notebook primarily reports accuracy.

Additional evaluation metrics would provide a more complete picture of model performance.

### Randomness in Some Models

The original Random Forest implementation does not specify a `random_state`, so repeated runs may produce slightly different results.

### Potential Real-Time Prediction Considerations

Some features may represent game-state information that is only known after a particular event occurs.

For a production real-time prediction system, feature availability at prediction time should be carefully defined.

---

# 🔧 Recommended Production Improvements

If this project is developed further into a production-ready machine learning application, the following architecture could be used:

```text
Raw CS:GO Game Data
        ↓
Data Validation
        ↓
Preprocessing Pipeline
        ↓
Feature Transformation
        ↓
Trained Random Forest Model
        ↓
Prediction API
        ↓
User Interface
        ↓
Predicted Round Winner
```

A Scikit-learn `Pipeline` could also be used to combine preprocessing and modeling into one reproducible workflow.

---

# 🗂️ Recommended GitHub Repository

A polished version of the repository could look like:

```text
CSGO-Round-Winner-Prediction/
│
├── data/
│   └── DataCGGO.csv
│
├── notebooks/
│   └── csgo_round_winner_prediction.ipynb
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

If the dataset is too large for normal GitHub repository storage, consider using Git LFS or an external dataset-hosting solution instead of committing the full CSV directly to the repository.

---

# 📦 requirements.txt

Create a file named:

```text
requirements.txt
```

with:

```text
pandas
numpy
seaborn
matplotlib
scikit-learn
jupyter
```

Then install everything using:

```bash
pip install -r requirements.txt
```

---

# 🚫 .gitignore

A basic `.gitignore` can contain:

```text
__pycache__/
*.py[cod]
.ipynb_checkpoints/
venv/
.env
.DS_Store
```

If you decide not to commit the large dataset to GitHub, you can additionally add:

```text
data/
```

and provide instructions for downloading the dataset separately.

---

# 🧑‍💻 Author

**Rakshith S**

Data Science / Machine Learning Project

GitHub:

```text
https://github.com/<YOUR-GITHUB-USERNAME>
```

LinkedIn:

```text
https://www.linkedin.com/in/<YOUR-LINKEDIN-USERNAME>
```

---

# ⭐ If You Find This Project Useful

If this project helped you understand machine learning classification or CS:GO game-state prediction, consider giving the repository a ⭐ on GitHub.

---

# 📜 License

This project can be released under the MIT License if you choose to make it open source.

Example:

```text
MIT License

Copyright (c) 2026 Rakshith S

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, subject to the conditions of the MIT License.
```

---

# ✅ Conclusion

This project demonstrates an end-to-end machine learning classification workflow using CS:GO game-state data.

The analysis begins with exploratory data analysis and data-quality checks, followed by categorical encoding, feature/target separation, train-test splitting, standardization, and Linear Discriminant Analysis.

Four classification algorithms are then evaluated:

```text
Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors
```

The recorded results are:

```text
Logistic Regression  → 74.81%
Decision Tree        → 82.37%
KNN                  → 83.33%
Random Forest        → 87.66%
```

Among the tested models, **Random Forest achieved the highest recorded accuracy of approximately 87.66%**.

The project provides a foundation for further development into a more advanced CS:GO round prediction system with cross-validation, hyperparameter tuning, feature importance analysis, additional evaluation metrics, and an interactive prediction application.
