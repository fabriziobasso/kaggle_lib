# Import Libraries:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import math

try: #Prevent the casting of an error message if the library is imported outside Kaggle/Colab
    from google.colab import drive
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

##################### FUNCTIONS ##################### 
# Connect to gdrive
def mount_drive(competition_name:str="", tp:str="/gdrive/MyDrive/Exercises/Studies_Structured_Data/Data/"):
  """
    Changes the current working directory to the specified path.
    Args:
        target_path (str): The folder path you want to move into.
  """
  
  if not COLAB_AVAILABLE:
        print("Google Colab environment not detected. Skipping drive mount.")
        target_path = rf"{tp}{competition_name}"
    
        try:
            # Step 1: Display the directory before the change
            print(f"Current directory before change: {os.getcwd()}")
          
            # Step 2: Change the working directory
            os.chdir(target_path)
          
            # Step 3: Display the directory after the change
            print(f"New working directory: {os.getcwd()}")
          
        except FileNotFoundError:
            print(f"Error: The system cannot find the path specified: '{target_path}'")
        except PermissionError:
            print(f"Error: You do not have permission to access: '{target_path}'")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
      
        return
  
  drive.mount(r'/gdrive')

  target_path = rf"{tp}{competition_name}"

  try:
      # Step 1: Display the directory before the change
      print(f"Current directory before change: {os.getcwd()}")
      
      # Step 2: Change the working directory
      os.chdir(target_path)
      
      # Step 3: Display the directory after the change
      print(f"New working directory: {os.getcwd()}")
      
  except FileNotFoundError:
      print(f"Error: The system cannot find the path specified: '{target_path}'")
  except PermissionError:
      print(f"Error: You do not have permission to access: '{target_path}'")
  except Exception as e:
      print(f"An unexpected error occurred: {e}")

def import_files(original_data:str="", datatypes: dict={}, mute=False, 
                 trn:str="train.csv", tst:str="test.csv", sub:str="sample_submission.csv"):
  
  """
  import files from drive:

  train=pd.read_csv("train.csv")
  test=pd.read_csv("test.csv")
  orig_data=pd.read_csv(original_data)
  submission=pd.read_csv("sample_submission.csv")
  """

  train=pd.read_csv(trn, index_col=0, dtype=datatypes)
  test=pd.read_csv(tst, index_col=0, dtype=datatypes)
  if original_data!="":
    orig_data=pd.read_csv(original_data, dtype=datatypes)
  
  submission=pd.read_csv(sub, index_col=0)
                     
  print(f"Train shape: {train.shape}")
  print(f"Test shape: {test.shape}")
  print(f"Submission shape: {submission.shape}")
  if original_data!="":
    print(f"Original data shape: {orig_data.shape}")

  if mute==False:
    print(f"Train shape: {train.info()}")
    print(f"Test shape: {test.info()}")
    if original_data!="":
      print(f"Original data shape: {orig_data.info()}")
    print(f"Submission shape: {submission.info()}")
  
  if original_data!="":
    return train, test, orig_data, submission
  else:
    return train, test, submission

def concatenate_train_original(train, orig_data):
  """
  Concatenate the train and original data.
  An additional Feature "Source" to identify the original data.

  Args:
      train (pd.DataFrame): The DataFrame containing the train data.
      orig_data (pd.DataFrame): The DataFrame containing the original data.

  Returns:
      pd.DataFrame: A new DataFrame containing the merged data.
  """

  train["Source"] = "Train"
  orig_data["Source"] = "Original"

  return pd.concat([train, orig_data], ignore_index=True)

def get_optimal_dtypes(file_path, nrows=1000):
    """
    Analyzes a sample of a dataset to determine memory-efficient datatypes.
    
    Args:
        file_path (str): Path to the CSV or Excel file.
        nrows (int): Number of rows to sample for analysis.
        
    Returns:
        dict: A mapping of column names to their recommended pandas dtypes.
    """
    # Step 1: Ingest the sample
    if file_path.endswith('.csv'):
        sample = pd.read_csv(file_path, nrows=nrows)
    elif file_path.endswith(('.xls', '.xlsx')):
        sample = pd.read_excel(file_path, nrows=nrows)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")

    optimal_dtypes = {}

    for col in sample.columns:
        # Check if the column is purely numeric
        if pd.api.types.is_numeric_dtype(sample[col]):
            if pd.api.types.is_float_dtype(sample[col]):
                optimal_dtypes[col] = 'float32'
            else:
                # Downcast integers based on min/max
                c_min, c_max = sample[col].min(), sample[col].max()
                if c_min >= 0:
                    if c_max < 255: optimal_dtypes[col] = 'uint8'
                    elif c_max < 65535: optimal_dtypes[col] = 'uint16'
                    else: optimal_dtypes[col] = 'uint32'
                else:
                    if c_min > -128 and c_max < 127: optimal_dtypes[col] = 'int8'
                    else: optimal_dtypes[col] = 'int32'
        
        # 2. Handle everything else (Strings/Objects)
        else:
            # We use 'category' for strings during ingestion. 
            # This avoids casting errors and is very memory efficient.
            optimal_dtypes[col] = 'category'
                
    return optimal_dtypes

#classify Features by datatypes:
def classify_features_by_dtype(df):
    """
    Classifies the columns of a DataFrame based on their data types.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to analyze.
    
    Returns:
    dict: A dictionary where keys are data types (strings) and 
          values are lists of column names belonging to that type.
    """
    # Initialize an empty dictionary to store our results
    dtype_dict = {}
    
    # Iterate through each unique data type found in the DataFrame columns
    for dtype in df.dtypes.unique():
        # df.select_dtypes returns a subset of the DataFrame with the chosen type
        # We then extract the column names as a list
        columns = df.select_dtypes(include=[dtype]).columns.tolist()
        
        # Store the list in the dictionary using the string name of the dtype
        dtype_dict[str(dtype)] = columns
        
    return dtype_dict

# Graph Categorical in countplot
def plot_categorical_percentages(df, features, hue_var):
    """
    Creates a grid of grouped bar charts showing percentages.
    Requires Seaborn v0.13.0 or higher.
    """
    # 1. Grid Configuration
    num_features = len(features)
    cols = 2
    rows = math.ceil(num_features / cols)
    
    # 2. Initialize Figure
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    
    # Ensure axes is an array even for a single plot
    if num_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    # 3. Plotting Loop
    for i, feature in enumerate(features):
        sns.countplot(
            data=df, 
            x=feature, 
            hue=hue_var, 
            stat="percent",      # Automatically converts counts to percent
            ax=axes[i]
        )
        
        # Formatting titles and labels
        axes[i].set_title(f"Distribution of {feature} by {hue_var}", fontsize=14)
        axes[i].set_ylabel("Percentage (%)")
        axes[i].set_xlabel(feature)

        # --- Visual Styling Start ---    
        # Add horizontal grid lines with dashed format
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        # Ensure grid lines are drawn behind the bars
        axes[i].set_axisbelow(True)
        # Remove top and right spines
        sns.despine(ax=axes[i], top=True, right=True)
        # --- Visual Styling End ---

    # 4. Cleanup: Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

def plot_categorical_percentages_hist(df, features, hue_var):
    """
    Creates a grid of grouped bar charts showing percentages.
    Uses sns.histplot to support common_norm on categorical data.
    """
    # 1. Determine Grid Layout
    num_features = len(features)
    cols = 2
    rows = math.ceil(num_features / cols)
    
    # 2. Initialize the Figure
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    
    # Flatten axes for easy iteration (handles single or multiple plots)
    if num_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    # 3. Plotting Loop
    for i, feature in enumerate(features):
        sns.histplot(
            data=df,
            x=feature,
            hue=hue_var,
            stat="percent",
            common_norm=False,  # This now works in histplot
            multiple="dodge",   # Makes it a grouped bar chart
            discrete=True,      # Required for categorical/discrete x-axis
            shrink=0.8,         # Adds spacing between bar groups
            ax=axes[i]
        )
        
        # Formatting titles and labels
        axes[i].set_title(f"Percentage Distribution: {feature} by {hue_var}", fontsize=14)
        axes[i].set_ylabel("Percentage (%)")
        axes[i].set_xlabel(feature)

        # --- Visual Styling Start ---    
        # Add horizontal grid lines with dashed format
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        # Ensure grid lines are drawn behind the bars
        axes[i].set_axisbelow(True)
        # Remove top and right spines
        sns.despine(ax=axes[i], top=True, right=True)
        # --- Visual Styling End ---

    # 4. Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

# Mutual Inforamtion analysis
def analyze_categorical_mi(df, features, target, random_state=42, graph=True):
    """
    Performs Mutual Information analysis on categorical features.
    
    Parameters:
    - df: The Pandas DataFrame.
    - features: List of strings (column names) to analyze.
    - target: String, the name of the target classification column.
    
    Returns:
    - mi_df: A DataFrame containing features and their MI scores.
    """
    
    # 1. Create a copy to avoid modifying the original dataframe
    X = df[features].copy()
    y = df[target]
    
    # 2. Encode categorical features and target to integers
    # mutual_info_classif requires numeric input, but we will 
    # flag them as discrete.
    le = LabelEncoder()
    
    for col in features:
        X[col] = le.fit_transform(X[col].astype(str))
        
    if y.dtype == 'object' or str(y.dtype) == 'category':
        y_encoded = le.fit_transform(y.astype(str))
    else:
        y_encoded = y

    # 3. Calculate Mutual Information
    # discrete_features=True is CRITICAL for purely categorical data
    mi_scores = mutual_info_classif(
        X, y_encoded, 
        discrete_features=True, 
        random_state=random_state
    )
    
    # 4. Create the summary DataFrame
    mi_df = pd.DataFrame({
        'Feature': features,
        'MI_Score': mi_scores
    }).sort_values(by='MI_Score', ascending=False).reset_index(drop=True)
    
    # 5. Plot the results
    if graph==True:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        sns.barplot(
            x='MI_Score', 
            y='Feature', 
            data=mi_df, 
            hue='Feature', 
            palette='RdYlBu', 
            legend=False
        )
        
        plt.title(f'Mutual Information Scores (Target: {target})', fontsize=14)
        plt.xlabel('Information Gain / MI Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        sns.despine(top=True, right=True)
        plt.tight_layout()
        plt.show()
    
    return mi_df

# Encode Categorical Features:
def encode_categorical_features(train_df, test_df, cat_features):
    """
    Fits an OrdinalEncoder on the training dataset and applies the 
    transformation to both the training and test datasets.
    
    Parameters:
    - train_df: The training DataFrame.
    - test_df: The test DataFrame.
    - cat_features: List of column names (strings) to encode.
    
    Returns:
    - train_encoded: Encoded training DataFrame.
    - test_encoded: Encoded test DataFrame.
    - encoder: The fitted OrdinalEncoder object.
    """
    # 1. Create copies to avoid side effects on the original dataframes
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # 2. Initialize the encoder
    # handle_unknown='use_encoded_value' prevents errors if the test set 
    # contains a category not found in the training set.
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', 
        unknown_value=-1
    )
    
    # 3. Fit on train and transform both
    # Ensure columns are cast to string to handle mixed types or NaNs gracefully
    train_encoded[cat_features] = encoder.fit_transform(train_encoded[cat_features].astype(str))
    test_encoded[cat_features] = encoder.transform(test_encoded[cat_features].astype(str))
    
    return train_encoded, test_encoded, encoder

# Rel with cont fratures:
def analyze_feature_relationships(df, features, target, problem_type='regression'):
    """
    Generates plots to visualize the relationship between features and a target variable.
    Updated to comply with Seaborn v0.14.0 palette requirements.
    
    Parameters:
    df (pd.DataFrame): The dataset containing features and target.
    features (list): A list of column names (strings) to analyze.
    target (str): The name of the target variable column.
    problem_type (str): Either 'regression' or 'classification'.
    """
    
    num_features = len(features)
    num_cols = 2
    num_rows = math.ceil(num_features / num_cols)
    
    sns.set_theme(style="whitegrid")
    
    # Create the figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows))
    
    # Ensure axes is an array even if there is only one plot
    if num_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(features):
        if problem_type.lower() == 'regression':
            # Scatter plot remains the same as it doesn't typically use a palette this way
            sns.scatterplot(data=df, x=col, y=target, ax=axes[i], color='teal')
            axes[i].set_title(f'Regression: {col} vs {target}')
            
        elif problem_type.lower() == 'classification':
            # FIX: Added hue=target and legend=False to address the FutureWarning
            sns.boxplot(
                data=df, 
                x=target, 
                y=col, 
                ax=axes[i], 
                hue=target, 
                legend=False, 
                palette='Set2'
            )
            axes[i].set_title(f'Classification: {col} by {target}')
            
        else:
            print(f"Unknown problem type: {problem_type}.")
            return

    # Clean up empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

# Correlation pairplot

def plot_feature_pairgrid(df, features, target, sample=1.0):
    """
    Creates a Seaborn pairplot to visualize relationships between features
    and a target variable.

    Parameters:
    df (pd.DataFrame): The input dataset.
    features (list): A list of strings representing the feature columns.
    target (str): The column name to use for color mapping (hue).
    """

    # We combine features and target into one list to ensure the pairplot
    # has access to the target variable for the 'hue' parameter.
    plot_columns = features + [target]

    # sample df
    df_ = df.sample(frac=sample)

    # Set the visual style
    sns.set_theme(style="ticks")

    # Generate the pairplot
    # vars: limits the variables plotted to our list
    # hue: colors the points based on the target categories
    # corner: removes the redundant upper triangle of the grid
    pair_grid = sns.pairplot(
        df_[plot_columns],
        vars=features,
        hue=target,
        corner=True,
        palette='Set3'
    )

    # Add a title to the figure
    pair_grid.fig.suptitle(f'Pairwise Relationship of Features (Hue: {target})', y=1.02)

    plt.show()
