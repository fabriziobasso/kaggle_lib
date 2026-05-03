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

def import_files(original_data:str="", mute=False):
  
  """
  import files from drive:

  train=pd.read_csv("train.csv")
  test=pd.read_csv("test.csv")
  orig_data=pd.read_csv(original_data)
  submission=pd.read_csv("sample_submission.csv")
  """

  train=pd.read_csv("train.csv", index_col=0)
  test=pd.read_csv("test.csv", index_col=0)
  if original_data!="":
    orig_data=pd.read_csv(original_data)
  
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
