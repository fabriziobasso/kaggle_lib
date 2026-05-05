# Import Libraries:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    
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
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    
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
