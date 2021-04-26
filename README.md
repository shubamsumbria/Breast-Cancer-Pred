# Comparative Analysis of Different Machine Learning Classification Algorithms for Breast Cancer Prediction
### In Dev!

This is a Python- based implementation of Different Classification Algorithms on the task of Breast Cancer Prediction using Machine Learning.
## About Dataset:
Breast Cancer Wisconsin (Diagnostic) Data Set from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

#### [Data Set Information](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)#:~:text=Data%20Set%20Information%3A,cd%20math%2Dprog%2Fcpo%2Ddataset%2Fmachine%2Dlearn%2FWDBC%2F)

#### Attribute Information:
1. `ID number`
2. `Diagnosis` (M = malignant, B = benign)
##### Ten real-valued features are computed for each cell nucleus(3-32):
- `radius` (mean of distances from center to points on the perimeter)
- `texture` (standard deviation of gray-scale values)
- `perimeter`
- `area`
- `smoothness` (local variation in radius lengths)
- `compactness` (perimeter^2 / area - 1.0)
- `concavity` (severity of concave portions of the contour)
- `concave points` (number of concave portions of the contour)
- `symmetry`
- `fractal dimension` ("coastline approximation" - 1)

#### Creators:
1. Dr. William H. Wolberg, General Surgery Dept.
University of Wisconsin, Clinical Sciences Center
Madison, WI 53792
wolberg '@' eagle.surgery.wisc.edu
2. W. Nick Street, Computer Sciences Dept.
University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
street '@' cs.wisc.edu 608-262-6619
3. Olvi L. Mangasarian, Computer Sciences Dept.
University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
olvi '@' cs.wisc.edu
#### Donor:
Nick Street

### Libraries Used:

<table>
<tbody>
<tr>
<td><a><img src="https://pandas.pydata.org/docs/_static/pandas.svg" alt="Seaborn" align="center" width="75"/></a></td>
<td><a><img src="https://matplotlib.org/_static/logo2_compressed.svg" alt="cplusplus" align="center" width="75"/></a></td>
<td><a><img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" alt="Seaborn" align="center" width="75"/></a></td>
<td> <a><img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" align="center" width="75"/></a></td>
</tr>
</tbody>
</table>

## Exploratory Data Analysis

### Checking Null and Missing Values
```
Null Values:
 diagnosis                  0
radius_mean                0
texture_mean               0
perimeter_mean             0
area_mean                  0
smoothness_mean            0
compactness_mean           0
concavity_mean             0
concave points_mean        0
symmetry_mean              0
fractal_dimension_mean     0
radius_se                  0
texture_se                 0
perimeter_se               0
area_se                    0
smoothness_se              0
compactness_se             0
concavity_se               0
concave points_se          0
symmetry_se                0
fractal_dimension_se       0
radius_worst               0
texture_worst              0
perimeter_worst            0
area_worst                 0
smoothness_worst           0
compactness_worst          0
concavity_worst            0
concave points_worst       0
symmetry_worst             0
fractal_dimension_worst    0
dtype: int64

Missing Values:
 diagnosis                  0
radius_mean                0
texture_mean               0
perimeter_mean             0
area_mean                  0
smoothness_mean            0
compactness_mean           0
concavity_mean             0
concave points_mean        0
symmetry_mean              0
fractal_dimension_mean     0
radius_se                  0
texture_se                 0
perimeter_se               0
area_se                    0
smoothness_se              0
compactness_se             0
concavity_se               0
concave points_se          0
symmetry_se                0
fractal_dimension_se       0
radius_worst               0
texture_worst              0
perimeter_worst            0
area_worst                 0
smoothness_worst           0
compactness_worst          0
concavity_worst            0
concave points_worst       0
symmetry_worst             0
fractal_dimension_worst    0
dtype: int64
```
- After checking various aspects like null values count, missing values count, and info. This dataset is perfect because of no Nul and missing values.

### Information of dataset

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 569 entries, 0 to 568
Data columns (total 31 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   diagnosis                569 non-null    int64  
 1   radius_mean              569 non-null    float64
 2   texture_mean             569 non-null    float64
 3   perimeter_mean           569 non-null    float64
 4   area_mean                569 non-null    float64
 5   smoothness_mean          569 non-null    float64
 6   compactness_mean         569 non-null    float64
 7   concavity_mean           569 non-null    float64
 8   concave points_mean      569 non-null    float64
 9   symmetry_mean            569 non-null    float64
 10  fractal_dimension_mean   569 non-null    float64
 11  radius_se                569 non-null    float64
 12  texture_se               569 non-null    float64
 13  perimeter_se             569 non-null    float64
 14  area_se                  569 non-null    float64
 15  smoothness_se            569 non-null    float64
 16  compactness_se           569 non-null    float64
 17  concavity_se             569 non-null    float64
 18  concave points_se        569 non-null    float64
 19  symmetry_se              569 non-null    float64
 20  fractal_dimension_se     569 non-null    float64
 21  radius_worst             569 non-null    float64
 22  texture_worst            569 non-null    float64
 23  perimeter_worst          569 non-null    float64
 24  area_worst               569 non-null    float64
 25  smoothness_worst         569 non-null    float64
 26  compactness_worst        569 non-null    float64
 27  concavity_worst          569 non-null    float64
 28  concave points_worst     569 non-null    float64
 29  symmetry_worst           569 non-null    float64
 30  fractal_dimension_worst  569 non-null    float64
dtypes: float64(30), int64(1)
memory usage: 137.9 KB
```
### Statistical Description of Data

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/0.png" width="720"/></a>

### Count Based On Diagnosis:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/1.png" width="720"/></a>

**Observation:** We have 357 malignant cases and 212 benign cases so our dataset is Imbalanced, we can use various re-sampling algorithms like under-sampling, over-sampling, SMOTE, etc. Use the “adequate” correct algorithm.

### Correlation with Diagnosis:

#### Correlation of Mean Features with Diagnosis:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/2.png" width="720"/></a>

**Observations:**
- fractal_dimension_mean least correlated with the target variable.
- All other mean features have a significant correlation with the target variable.

#### Correlation of Squared Error Features with Diagnosis:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/3.png" width="720"/></a>

**Observations:**
- texture_se, smoothness_se, symmetry_se, and fractal_dimension_se are least correlated with the target variable.
- All other squared error features have a significant correlation with the target variable.

#### Correlation of Worst Features with Diagnosis:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/4.png" width="720"/></a>

- **Observation:** All worst features have a significant correlation with the target variable.

### Distribution based on Nucleus and Diagnosis:

#### Mean Features vs Diagnosis:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/5.png" width="720"/></a>

#### Squared Error Features vs Diagnosis:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/6.png" width="720"/></a>

#### Worst Features vs Diagnosis:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/7.png" width="720"/></a>

### Checking Multicollinearity Between Distinct Features:

#### Mean Features:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/8.png" width="720"/></a>

#### Squared Error Features:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/9.png" width="720"/></a>

#### Worst Features:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/10.png" width="720"/></a>

**Observations:**
- Almost perfectly linear patterns between the radius, perimeter, and area attributes are hinting at the presence of multicollinearity between these variables.
- Another set of variables that possibly imply multicollinearity are the concavity, concave_points, and compactness.

### Correlation Heatmap between Nucleus Feature:

<a><img src="https://github.com/shubamsumbria66/Breast-Cancer-Pred/blob/main/graphs/11.png" width="720"/></a>

- **Observations:** We can verify multicollinearity between some variables. This is because the three columns essentially contain the same information, which is the physical size of the observation (the cell). Therefore, we should only pick one of the three columns when we go into further analysis.

### Things to remember while working with this dataset:

- Slightly Imbalanced dataset (357 malignant cases and 212 benign cases). We have to select an adequate re-sampling algorithm for balancing.
- Multicollinearity between some features.
- As three columns essentially contain the same information, which is the physical size of the cell, we have to choose an appropriate feature selection method to eliminate unnecessary features.

## Classifiers Used:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors
5. Linear SVM
6. Kernal SVM
7. Gaussian Naive Bayes
