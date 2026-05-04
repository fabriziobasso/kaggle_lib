# Import Libraries:
import pandas as pd
import numpy as np
import os
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

def import_files(original_data:str="", datatypes: dict={}, mute=False):
  
  """
  import files from drive:

  train=pd.read_csv("train.csv")
  test=pd.read_csv("test.csv")
  orig_data=pd.read_csv(original_data)
  submission=pd.read_csv("sample_submission.csv")
  """

  train=pd.read_csv("train.csv", index_col=0, dtype=datatypes)
  test=pd.read_csv("test.csv", index_col=0, dtype=datatypes)
  if original_data!="":
    orig_data=pd.read_csv(original_data, dtype=datatypes)
  
  submission=pd.read_csv("sample_submission.csv", index_col=0)
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
