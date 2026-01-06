Perform end-to-end analysis on a real-world dataset using Python (Pandas, NumPy, Matplotlib, Seaborn, scikit-learn). The tasks include:

#1.use Classification Model Example:iris dataset
#step1.Data loading and inspection
import seaborn as sns
df=sns.load_dataset("iris")
print(df)
# First 5 row
print(df.head())
# Dataset info
print(df.info())
# Shape
print(df.shape)

#step2.Data cleaning and preprocessing
# Check missing values
print(df.isnull().sum())
# Check missing values
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# After cleaning
print(df.shape)
import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Heidi'],
   'Age': [24, 30, np.nan, 28, 22, 35, 29, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'New                                 York',  'Boston', 'Los Angeles',  'Chicago', ''],
    'Salary': [50000, 60000, 55000, 62000, np.nan, 70000, 58000, 75000],
    'Department': ['IT', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'IT']
}
df=pd.DataFrame(data)
# Check missing values
print(df.isnull().sum())
#Handling the missing values
# Fill missing Age with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Salary with 0
df['Salary'].fillna(0, inplace=True)
df = df[df['City'] != ""]
print("\nDataFrame after handling missing values:")
print(df)
#Handle Duplicates
df_dup=pd.DataFrame({'A':[1,2,2,3],'B':['x','y','y','z']})
print (df_dup)
#Remove the duplicates
df_dup.drop_duplicates(inplace=True)
print(df_dup)
df['Age']=df['Age'].astype(int)
print(df.dtypes)
#Data Type conversion
df['Age']=df['Age'].astype(int)
print (df.dtypes)

#step3.Exploratory Data Analysis(EDA)
df=sns.load_dataset("iris")
print ("\nSummary Statistics:")
print (df.describe())
#Check the distribution using Histogram
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib .pyplot as plt
data=np.random.randn(1000)
plt.hist(data, bins= 30,color="green",edgecolor="black")
plt.title("Histogram example")
plt.xlabel("values")
plt.ylabel("frequency")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
numeric_df = df.select_dtypes(include='number')

# correlation calculate
corr = numeric_df.corr()
sns.heatmap(corr, annot=True,     cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Step4.Data Visualization
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
df = pd.DataFrame({
    'Age': [18, 19, 20, 21, 22],
    'Marks': [65, 70, 75, 80, 85]
})

# Line plot (Matplotlib)
plt.plot(df['Age'], df['Marks'])
plt.xlabel("Age")
plt.ylabel("Marks")
plt.title("Line Plot")
plt.show()

# bar plot
categories = ['A', 'B', 'C', 'D']
values = [10, 24, 36, 18]
plt.bar(categories, values, color="Pink")
plt.title("Bar Chart Example")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.legend()
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("iris")
# .Scatter Plot (Seaborn)
sns.scatterplot(
    x="sepal_length",
    y="sepal_width",
    hue="species",
    data=df
)
plt.title("Scatter Plot - Iris Dataset")
plt.show()
# Box Plot (Seaborn)
sns.boxplot(
    x="species",
    y="petal_length",
    data=df,
    palette="Set2"
)
plt.title("Box Plot - Petal Length by Species")
plt.show()

#  Pair Plot (Seaborn)
sns.pairplot(df, hue="species")
plt.suptitle("Pair Plot - Iris Dataset", y=1.02)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("iris")

# Violin Plot
sns.violinplot(
    x="species",
    y="petal_length",
    data=df,
    palette="Set2")
plt.title("Violin Plot - Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length")
plt.show()

#step5: Building a predictive Model 
import seaborn as sns
from sklearn.linear_model
 import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection
 import train_test_split

# Load dataset
df = sns.load_dataset("iris")

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

#step6.Evaluation
print("\nClassification Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",      classification_report(y_test, y_pred))