#Perform end-to-end analysis on a real-world dataset using Python (Pandas, NumPy,
Matplotlib, Seaborn, scikit-learn).
#1.Use Classification Model Example : iris dataset
#step 1 - 1: Data loading and inspection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
# Load the dataset
df = sns.load_dataset("iris")
print(df)
# Inspection
print("First 5 rows of dataset:\n",df.head())
print("\nDataset Info:")
print(df.info())
#Step-2-Data Cleaning and Preprocessing
# Check for missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())
# Handle Duplicates (if any)
print("DataFrame with duplicates:")
print(df)
df.drop_duplicates(inplace=True)
print("\nDataFrame after dropping duplicates:")
print(df)
#Converting datatypes using astype function
df['species'] = df['species'].astype(str)
print(df)
#Step-3 - EDA
#Use describe() to get statistical summary
print("\nSummary Statistics:")
print(df.describe())
#Check distributions using histograms or boxplots
sns.boxplot(x="species", y="petal_length",data=df, palette="Set2")
plt.title("Box Plot - Petal Length bySpecies")
plt.show()
numeric_df=df.select_dtypes(include='number')
corr=numeric_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
# 1. Identify Numerical Variables
# These are usually 'float64' or 'int64'
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
# 2. Identify Categorical Variables
# These are usually 'object'
,
'category'
, or 'bool'
categorical_cols
df.select_dtypes(include=['object''category']).columns.tolist()
print(f"Numerical Variables:{numerical_cols}")
print(f"Categorical Variables: {categorical_cols}")
# Quick check of data types
print("\nColumn Data Types:")
print(df.dtypes)
#Step-4 - Data Visualization
#Line plot
iris = load_iris()
x = range(len(iris.data))
sepal_length = iris.data[:, 0]
sepal_width = iris.data[:, 1]
petal_length = iris.data[:, 2]
plt.figure()
plt.plot(x, sepal_length, label='Sepal Length')
plt.plot(x, sepal_width, label='Sepal Width')
plt.plot(x, petal_length, label='PetaLength')
plt.xlabel('Samples')
plt.ylabel('Length (cm)')
plt.title('Line Plot of Iris Features')
plt.legend()
plt.show()
#Scatter chart
plt.figure()
plt.scatter(sepal_length, sepal_width)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter Plot: Sepal Length vs Sepal Width')
plt.show()
#Bar Chart
avg_sepal_length = sepal_length.mean()
avg_sepal_width = sepal_width.mean()
avg_petal_length = petal_length.mean()
features = ['Sepal Length'
,
'Sepal Width'
,
'Petal Length']
values = [avg_sepal_length, avg_sepal_width, avg_petal_length]
plt.figure()
plt.bar(features, values)
plt.ylabel('Average Length (cm)')
plt.title('Bar Chart of Iris Features')
plt.show()
#Box plot
# Create a box plot for Petal Length across species
plt.figure(figsize=(10, 6))
sns.boxplot(x=
'species'
, y=
'petal_length'
, data=df, palette=
'Set2')
# Adding titles and labels
plt.title('Distribution of Petal Length by Species', fontsize=15)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Petal Length (cm)', fontsize=12)
plt.show()
# 6. Pair Plot (Seaborn)
sns.pairplot(df, hue=
"species")
plt.suptitle("Pair Plot - Iris Dataset", y=1.02)
plt.show()
#7.Violin plot
# 1. Load the dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]
# 2. Set the visual style
sns.set_theme(style="whitegrid")
# 3. Create the Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=
'species'
, y=
'petal length (cm)'
, data=df, inner=
'quartile'
,
palette=
'muted')
# 4. Adding titles and labels
plt.title('Distribution of Petal Length by Species (Violin Plot)'
, fontsize=15)
plt.xlabel('Species'
, fontsize=12)
plt.ylabel('Petal Length (cm)'
, fontsize=12)
plt.savefig('iris_violin_plot.png')
#Step-5 - Predictive modelling
# Import libraries
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load dataset
iris = sns.load_dataset("iris")
# Features and target
X = iris[['sepal_length'
,
'sepal_width'
,
'petal_length']]
y = iris['species']
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Choose Logistic Regression model
model = LogisticRegression(max_iter=200)
# Train the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Check accuracy
print("Accuracy:"
, accuracy_score(y_test, y_pred))
#Step-6 - Model Evaluation
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#1.Use Regression Model Example : salary dataset

# DATA LOADING
#Load dataset using pandas and seaborn
import pandas as pd
import seaborn as sns
df=pd.read_csv("Salary_Data.csv")
#Inspect the first few rows with df.head()
print("First 5 rows of dataset:\n",df.head())
#Check shape and column types with df.info()
print("\nDataset Info:")
print(df.info())
# step1:DATA CLEANING
#Checking missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())
df=df.dropna()
print(df.isnull().sum())
#EXPLORATORY DATA ANALYSIS (EDA)
print("\nSummary Statistics:")
print(df.describe())
# step2:DATA VISUALIZATION
# importing matplot library for graphs
import matplotlib.pyplot as plt
#Univariate Analysis
df.hist(figsize=(6,4))
plt.title("Histogram on salary dataset")
plt.show()
# step3:Bivariate Analysis
plt.scatter(df['Years of Experience'], df['Salary'])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.show()
#Correlation Heatmap
numeric_df=df.select_dtypes(include='number')
corr=numeric_df.corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
# step4:PREDICTIVE MODELING
#LINEAR REGRESSION
#Regression Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
# Features and target
X = df[['Years of Experience']]   # Independent variable
y = df['Salary']              # Dependent variable
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
# Predictions
y_pred = lin_reg.predict(X_test)
#step6: Evaluation
print("Linear Regression Results")
print("Intercept:", lin_reg.intercept_)
print("Coefficient:", lin_reg.coef_)
print("Mean Squared Error:",mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
print("Root Mean Squared Error:",np.sqrt(mse) )
print("R² Score:",r2_score(y_test, y_pred))