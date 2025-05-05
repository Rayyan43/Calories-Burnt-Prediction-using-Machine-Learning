#!/usr/bin/env python
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,OrdinalEncoder

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


# In[4]:

# 1. Reading data from CSV
def read_csv(file_path):
    """
    Read data from a CSV file and return a pandas DataFrame.

    Parameters:
    - file_path: str, the path to the CSV file.

    Returns:
    - pd.DataFrame, the loaded DataFrame.
    """
    return pd.read_csv(file_path)
#2. Getting information and statistics about over dataset
def dataset_info_statistics(data):
    """
    Display information and basic statistics about the dataset.

    Parameters:
    - data: pandas DataFrame, input data.

    Returns:
    - None
    """
    # Display general information about the dataset
    print("Dataset Information:")
    print(data.info())
    print("\n")

    # Display basic statistics for numerical columns
    print("Basic Statistics for Numerical Columns:")
    print(data.describe())
    print("\n")

#3.check for the null values in the dataset
def check_null(data):
    """
    Check for null values in the dataset.

    Parameters:
    - data: pandas DataFrame, input data.

    Returns:
    - pd.Series, the count of null values for each column.
    """
    null_counts = data.isnull().sum()
    print("Null Values in the Dataset:")
    return null_counts

#4.check for duplicated rows in the dataset
def check_duplicates(data):
    """
    Check for duplicated rows in the dataset.

    Parameters:
    - data: pandas DataFrame, input data.

    Returns:
    - bool, True if any duplicated rows exist, False otherwise.
    """
    return data.duplicated().any()

#5. getting basic analysis for numerical and categorical columns
def plot_graph(data):
    """
    Plot graphs for numerical and categorical data in a dataframe.
    
    Parameters:
    - data: Pandas Dataframe, input data.
    
    Returns:
    - None
    
    """
    numerical_columns = data.select_dtypes(include=np.number).columns
     
    for column in numerical_columns:
        plt.figure(figsize=(5,3))
        sns.distplot(data[column],kde=True)
        plt.title(f"Histogram for {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()
        
    categorical_columns = data.select_dtypes(include='object').columns
    for column in categorical_columns:
        plt.figure(figsize=(5, 3))
        sns.countplot(data[column])
        plt.title(f'Countplot for {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
    
#6. Seperate feature and target
def seperate_features_target(data,target_column):
    """
    Separate features and target variable
    
    Parameters: 
    - data: pandas DataFrame, input data.
    - target_column: str, the column representing the target varible.
    
    Returns:
    - X: pandas DataFrame, features.
    - y: pandas Series, target variable.
    
    """
    
    X = data.drop(columns=[target_column],axis=1)
    y = data[target_column]
    
    return X,y
#7. Train test split
def perform_train_test_split(X, y, test_size=0.20, random_state=42):
    """
    Perform train-test split on the dataset.

    Parameters:
    - X: pandas DataFrame, features.
    - y: pandas Series, target variable.
    - test_size: float, optional, the proportion of the dataset to include in the test split (default is 0.2).
    - random_state: int or None, optional, seed for random number generation (default is None).

    Returns:
    - X_train: pandas DataFrame, features for training.
    - X_test: pandas DataFrame, features for testing.
    - y_train: pandas Series, target variable for training.
    - y_test: pandas Series, target variable for testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# In[5]:


calories = read_csv('calories.csv')
exercise = read_csv('exercise.csv')


# In[6]:


data = pd.merge(calories, exercise, on='User_ID')


# In[7]:


data.head()


# In[8]:


dataset_info_statistics(data)


# In[9]:


check_null(data)


# In[10]:


#plot_graph(data)


# In[11]:


data.columns


# In[12]:


X,y = seperate_features_target(data,'Calories')


# In[13]:


X = X.drop(columns=['User_ID'])


# In[14]:


X_train,X_test,y_train,y_test = perform_train_test_split(X, y, test_size=0.20, random_state=42)


# ### Column Transformer and Pipeline

# In[16]:


preprocessor = ColumnTransformer(transformers=[
    ('ordinal',OrdinalEncoder(),['Gender']),
    ('num',StandardScaler(),['Age',
                            'Height',
                            'Weight',
                            'Duration',
                            'Heart_Rate',
                            'Body_Temp']),
],remainder='passthrough')


# In[17]:


pipeline = Pipeline([("preprocessor",preprocessor),
                     ("model",LinearRegression())
                    ])


# In[18]:


from sklearn import set_config


# In[19]:


set_config(display='diagram')


# In[20]:


pipeline


# In[21]:


pipeline.fit(X_train,y_train)


# In[22]:


y_pred = pipeline.predict(X_test)


# In[23]:


from sklearn.metrics import r2_score


# In[24]:


r2_score(y_test,y_pred)


# In[25]:


from sklearn.model_selection import KFold


# In[26]:


kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# In[27]:


from sklearn.model_selection import cross_val_score


# In[28]:


cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')


# In[29]:


cv_results.mean()


# In[30]:


from sklearn.metrics import mean_absolute_error


# In[31]:


mean_absolute_error(y_test,y_pred)


# In[32]:


def model_scorer(model_name,model):
    
    output=[]
   
    
    output.append(model_name)
    
    pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',model)])
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    
    pipeline.fit(X_train,y_train)
    
    y_pred = pipeline.predict(X_test)
    
    output.append(r2_score(y_test,y_pred))
    output.append(mean_absolute_error(y_test,y_pred))
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
    output.append(cv_results.mean())
    
    return output


# In[33]:


model_dict={
    'log':LinearRegression(),
    'RF':RandomForestRegressor(),
    'XGBR':XGBRegressor(),
}


# In[34]:


model_output=[]
for model_name,model in model_dict.items():
    model_output.append(model_scorer(model_name,model))


# In[35]:


model_output


# In[36]:


preprocessor = ColumnTransformer(transformers=[
    ('ordinal',OrdinalEncoder(),['Gender']),
    ('num',StandardScaler(),['Age',
                            'Height',
                            'Weight',
                            'Duration',
                            'Heart_Rate',
                            'Body_Temp']),
    
],remainder='passthrough')


# In[37]:


pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',XGBRegressor())
    
])


# In[38]:


pipeline.fit(X,y)


# In[39]:


sample = pd.DataFrame({
   'Gender':'male',
    'Age':68,
    'Height':190.0,
    'Weight':94.0,
    'Duration':29.0,
    'Heart_Rate':105.0,
    'Body_Temp':40.8,
},index=[0])


# In[40]:


pipeline.predict(sample)


# ### Save The Model

# In[42]:


import pickle


# In[43]:


with open('pipeline.pkl','wb') as f:
    pickle.dump(pipeline,f)


# In[44]:


with open('pipeline.pkl','rb') as f:
    pipeline_saved = pickle.load(f)


# In[45]:


result = pipeline_saved.predict(sample)


# In[46]:


result 


# ### GUI

# In[48]:
import pickle
import pandas as pd
from tkinter import *
import os
from tkinter import messagebox

def clear_fields():
    """
    Clear all input fields in the GUI.
    """
    e2.delete(0, END)
    e3.delete(0, END)
    e4.delete(0, END)
    e5.delete(0, END)
    e6.delete(0, END)
    e7.delete(0, END)
    clicked.set("male")  # Reset the dropdown to default value

def show_entry():
    try:
        # Load the pipeline model
        if not os.path.exists('pipeline.pkl'):
            messagebox.showerror("Error", "Model file 'pipeline.pkl' not found!")
            return
        
        with open('pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)

        # Collect user inputs
        p1 = clicked.get()
        p2 = e2.get()
        p3 = e3.get()
        p4 = e4.get()
        p5 = e5.get()
        p6 = e6.get()
        p7 = e7.get()

        # Validate inputs
        if not all([p1, p2, p3, p4, p5, p6, p7]):
            messagebox.showwarning("Input Error", "Please fill in all fields!")
            return

        try:
            sample = pd.DataFrame({
                'Gender': [p1],
                'Age': [float(p2)],
                'Height': [float(p3)],
                'Weight': [float(p4)],
                'Duration': [float(p5)],
                'Heart_Rate': [float(p6)],
                'Body_Temp': [float(p7)],
            }, index=[0])
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values!")
            return

        # Predict and display result
        result = pipeline.predict(sample)
        messagebox.showinfo("Prediction Result", f"Estimated Calories Burnt: {result[0]:.2f}")

        # Clear the fields after displaying the result
        clear_fields()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the main window
master = Tk()
master.title("Calories Burnt Prediction")
master.geometry("400x300")
master.resizable(False, False)

# Add labels and input fields
Label(master, text="Calories Burnt Prediction", bg="black", fg="white", font=("Arial", 14)).grid(row=0, columnspan=2, pady=10)

Label(master, text="Select Gender").grid(row=1, sticky=W, padx=10)
Label(master, text="Enter Your Age").grid(row=2, sticky=W, padx=10)
Label(master, text="Enter Your Height (CM)").grid(row=3, sticky=W, padx=10)
Label(master, text="Enter Your Weight (lb)").grid(row=4, sticky=W, padx=10)
Label(master, text="Duration (min)").grid(row=5, sticky=W, padx=10)
Label(master, text="Heart Rate").grid(row=6, sticky=W, padx=10)
Label(master, text="Body Temp").grid(row=7, sticky=W, padx=10)

clicked = StringVar()
clicked.set("male")
options = ['male', 'female']

e1 = OptionMenu(master, clicked, *options)
e1.configure(width=15)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)

e1.grid(row=1, column=1, pady=5)
e2.grid(row=2, column=1, pady=5)
e3.grid(row=3, column=1, pady=5)
e4.grid(row=4, column=1, pady=5)
e5.grid(row=5, column=1, pady=5)
e6.grid(row=6, column=1, pady=5)
e7.grid(row=7, column=1, pady=5)

# Add predict button (centered)
Button(master, text="Predict", command=show_entry, bg="blue", fg="white").grid(row=8, column=0, columnspan=2, pady=10)

# Add clear button (right-aligned)
Button(master, text="Clear", command=clear_fields, bg="red", fg="white").grid(row=9, column=1, pady=10)

mainloop()