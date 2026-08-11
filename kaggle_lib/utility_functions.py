# Import Libraries:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from itertools import combinations
import math
from tqdm import tqdm


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer,    
                                    MaxAbsScaler,
                                    MinMaxScaler,
                                    Normalizer,
                                    PowerTransformer,
                                    QuantileTransformer,
                                    RobustScaler,
                                    LabelEncoder,
                                    OrdinalEncoder,
                                    StandardScaler,
                                    minmax_scale)

from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.base import clone

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
        palette='Set2'
    )

    # Add a title to the figure
    pair_grid.fig.suptitle(f'Pairwise Relationship of Features (Hue: {target})', y=1.02)

    plt.show()

# Visualize categorical features vs continous features with hue:

def plot_categorical_distributions(df, cat_features, target, hue=None, palette='viridis', fig_width=12, row_height=5):
    """
    Visualizes the distribution of a numerical target across categorical features,
    optionally split by a second categorical 'hue' feature.
    
    Parameters:
    df (pd.DataFrame): The dataset.
    cat_features (list): Categorical columns for the x-axis.
    target (str): Numerical column for the y-axis.
    hue (str, optional): Categorical column to split the boxplots (default: None).
    palette (str): Color palette name.
    fig_width (int): Width of the figure.
    row_height (int): Height per row of plots.
    """
    
    num_features = len(cat_features)
    num_cols = 2
    num_rows = math.ceil(num_features / num_cols)
    
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(
        num_rows, 
        num_cols, 
        figsize=(fig_width, row_height * num_rows)
    )
    
    # Standardize axes to a list for iteration
    if num_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, col in enumerate(cat_features):
        # Determine mapping: 
        # If hue is provided, we use it for color and show the legend.
        # If not, we use the x-axis variable for color to avoid warnings.
        current_hue = hue if hue else col
        show_legend = True if hue else False
        
        sns.boxplot(
            data=df, 
            x=col, 
            y=target, 
            hue=current_hue,
            ax=axes[i], 
            palette=palette,
            legend=show_legend
        )
        
        # Titles and Labels
        title_suffix = f" (split by {hue})" if hue else ""
        axes[i].set_title(f'{target} by {col}{title_suffix}', fontsize=12, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

# Create pipeline to transform data:
def build_preprocessing_pipeline(transformation_dict):
    """
    Constructs an unfitted sklearn ColumnTransformer pipeline based on a 
    dictionary of features and their corresponding transformation steps.
    
    Parameters:
    transformation_dict (dict): Keys are feature names (str), values are lists 
                                of transformations (functions or sklearn objects).
                                
    Returns:
    sklearn.compose.ColumnTransformer: The modular pipeline object.
    """
    
    transformers = []
    
    # Iterate through the dictionary to build sub-pipelines for each feature
    for feature, steps in transformation_dict.items():
        pipeline_steps = []
        
        for i, step in enumerate(steps):
            # Step detection: If it's a function (np.log1p), wrap in FunctionTransformer
            if callable(step) and not hasattr(step, "fit_transform"):
                transformer = FunctionTransformer(step, validate=False)
            else:
                # If it's a class type (e.g., StandardScaler), instantiate it
                # If it's an instance, clone it to ensure a fresh state
                transformer = step() if isinstance(step, type) else clone(step)
            
            # Label each step numerically (step_0, step_1, etc.)
            pipeline_steps.append((f'step_{i}', transformer))
        
        # Create a Pipeline for this specific column
        feature_pipeline = Pipeline(pipeline_steps)
        
        # Add to the list of transformers for ColumnTransformer
        # Format: (name, pipeline_object, column_name)
        transformers.append((f'pipe_{feature}', feature_pipeline, [feature]))
    
    # Assemble the final ColumnTransformer
    # remainder='passthrough' keeps columns not defined in the dictionary
    preprocessing_pipeline = ColumnTransformer(
        transformers=transformers, 
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    # Ensure the output is a pandas DataFrame
    preprocessing_pipeline.set_output(transform="pandas")
    
    return preprocessing_pipeline

## Feature Engineering - Interaction Function for exploration
def rank_feature_interactions(df, target_feat, feature_list, problem_type='regression'):
    """
    Evaluates pairwise feature combinations and returns a ranked summary
    while avoiding DataFrame fragmentation.

    Parameters:
    df (pd.DataFrame): The source dataset.
    target_col (str): The column name of the target variable.
    feature_list (list): The list of features to combine.
    problem_type (str): 'regression' or 'classification'.

    Returns:
    pd.DataFrame: Ranked MI scores for each tested combination.
    """

    y = target_feat.copy()
    epsilon = 1e-9  # Handling values near zero

    # We use a dictionary to store new features to avoid fragmentation
    feature_dict = {}

    # 1. Generate all pairwise combinations
    for col_a, col_b in tqdm(combinations(feature_list, 2)):
        a, b = df[col_a], df[col_b]

        # Store each operation as a Series in our dictionary
        feature_dict[f"{col_a} + {col_b}"] = a + b
        feature_dict[f"{col_a} - {col_b}"] = a - b
        feature_dict[f"{col_a} * {col_b}"] = a * b
        feature_dict[f"{col_a} / {col_b}"] = a / (b + epsilon)
        feature_dict[f"{col_b} / {col_a}"] = b / (a + epsilon)

    # 2. Convert the dictionary to a DataFrame all at once
    # This is the key step to prevent PerformanceWarning
    temp_features = pd.concat(feature_dict, axis=1)

    # 3. Calculate Mutual Information
    if problem_type.lower() == 'regression':
        mi_scores = mutual_info_regression(temp_features, y)
    else:
        mi_scores = mutual_info_classif(temp_features, y)

    # 4. Create and sort the results DataFrame
    mi_report = pd.DataFrame({
        'Mutual_Information_Ratio': mi_scores
    }, index=temp_features.columns)

    # Sort from greatest to smallest
    mi_report = mi_report.sort_values(by='Mutual_Information_Ratio', ascending=False)

    return mi_report
  
############################################################################################################
################################## FEATURE SELECTION FUNCTIONS #############################################
############################################################################################################

def _apply_correlation_post_filter(df_ranked, X, top_n, threshold):
    """
    Evaluates the top N features for multicollinearity. 
    If two features are highly correlated, the one with the worse consensus rank is dropped.
    """
    # Isolate the top N features based on consensus rank
    top_df = df_ranked.head(top_n).copy()
    top_features = top_df['Feature'].tolist()
    
    # Calculate the absolute correlation matrix for only these top features
    corr_matrix = X[top_features].corr().abs()
    
    # Isolate the upper triangle to avoid checking pairs twice or checking self-correlation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    features_to_drop = set()
    
    for feature_col in upper.columns:
        # Find any row features correlated with this column feature above the threshold
        correlated_rows = upper.index[upper[feature_col] > threshold].tolist()
        
        for feature_row in correlated_rows:
            # Retrieve their consensus ranks
            rank_col = top_df.loc[top_df['Feature'] == feature_col, 'Consensus_Rank'].values[0]
            rank_row = top_df.loc[top_df['Feature'] == feature_row, 'Consensus_Rank'].values[0]
            
            # The feature with the higher numerical value (worse rank) gets marked for deletion
            if rank_col > rank_row:
                features_to_drop.add(feature_col)
            else:
                features_to_drop.add(feature_row)
                
    # Filter the dataframe to remove the redundant features
    final_df = top_df[~top_df['Feature'].isin(features_to_drop)].reset_index(drop=True)
    
    print(f"Post-Filter: Evaluated top {top_n} features. Dropped {len(features_to_drop)} redundant features.")
    if features_to_drop:
        print(f"Dropped features: {list(features_to_drop)}")
        
    return final_df


def select_features_classification(X, y, top_n=20, corr_threshold=0.85):
    """
    Classification feature selection with embedded correlation post-filtering.
    """
    print("Running Classification Feature Selection...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # The Four Pillars
    f_stat, _ = f_classif(X, y)
    mi = mutual_info_classif(X, y)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    
    lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
    lr.fit(X_scaled, y)
    l1_imp = np.abs(lr.coef_).mean(axis=0) if len(lr.classes_) > 2 else np.abs(lr.coef_[0])
    
    # Compile and Rank
    df = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': f_stat,
        'MI_Score': mi,
        'RF_Importance': rf_imp,
        'L1_Importance': l1_imp
    })
    
    for col in ['F_Score', 'MI_Score', 'RF_Importance', 'L1_Importance']:
        df[f'Rank_{col.split("_")[0]}'] = df[col].rank(ascending=False)
        
    rank_cols = [c for c in df.columns if c.startswith('Rank_')]
    df['Consensus_Rank'] = df[rank_cols].mean(axis=1)
    df = df.sort_values('Consensus_Rank').reset_index(drop=True)
    
    # Apply Correlation Post-Filter
    final_df = _apply_correlation_post_filter(df, X, top_n=top_n, threshold=corr_threshold)
    
    _plot_consensus(final_df, "Classification")
    return final_df


def select_features_regression(X, y, top_n=20, corr_threshold=0.85):
    """
    Regression feature selection with embedded correlation post-filtering.
    """
    print("Running Regression Feature Selection...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # The Four Pillars
    f_stat, _ = f_regression(X, y)
    mi = mutual_info_regression(X, y)
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_scaled, y)
    l1_imp = np.abs(lasso.coef_)
    
    # Compile and Rank
    df = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': f_stat,
        'MI_Score': mi,
        'RF_Importance': rf_imp,
        'L1_Importance': l1_imp
    })
    
    for col in ['F_Score', 'MI_Score', 'RF_Importance', 'L1_Importance']:
        df[f'Rank_{col.split("_")[0]}'] = df[col].rank(ascending=False)
        
    rank_cols = [c for c in df.columns if c.startswith('Rank_')]
    df['Consensus_Rank'] = df[rank_cols].mean(axis=1)
    df = df.sort_values('Consensus_Rank').reset_index(drop=True)
    
    # Apply Correlation Post-Filter
    final_df = _apply_correlation_post_filter(df, X, top_n=top_n, threshold=corr_threshold)
    
    _plot_consensus(final_df, "Regression")
    return final_df


def _plot_consensus(df, task_type):
    """Helper function to plot the post-filtered results."""
    plt.figure(figsize=(10, 8))
    
    sns.barplot(
        data=df, 
        x='Consensus_Rank', 
        y='Feature', 
        hue='Feature',
        palette='viridis', 
        legend=False
    )
    
    plt.title(f'Top Uncorrelated Features by Consensus Rank ({task_type})', fontsize=14)
    plt.xlabel('Average Rank Across 4 Methods (Lower = More Significant)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



###############################################################################
######################## Graph and plot functions #############################
###############################################################################

def plot_streamlined_histogram(data, feature_name, bins='auto', color='steelblue'):
    """
    Plots a highly streamlined univariate histogram for a specific feature,
    minimizing chart junk while retaining a dotted horizontal grid.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    - feature_name: str, the name of the column to plot.
    - bins: int or str, number of bins or binning strategy (default 'auto').
    - color: str, hex code or name for the bar color.
    """
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot the histogram with thin white edges for visual separation
    sns.histplot(
        data=data,
        x=feature_name,
        bins=bins,
        color=color,
        edgecolor='white',
        alpha=0.85,
        ax=ax
    )

    # 1. Streamline Axes: Remove top, right, and left spines entirely
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Keep the bottom spine for grounding, but soften its color
    ax.spines['bottom'].set_color('#cccccc')

    # 2. Clean Ticks: Remove y-axis tick lines (keep the numbers)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', color='#cccccc')

    # 3. Handle the Grid: Add horizontal dotted grid lines
    # Setting zorder/axisbelow ensures the grid stays *behind* the histogram bars
    ax.grid(axis='y', linestyle=':', color='gray', alpha=0.7)
    ax.set_axisbelow(True)

    # 4. Clean Labels and Title
    plt.title(f'Distribution of {feature_name}', fontsize=14, pad=15, loc='left', fontweight='bold', color='#333333')
    plt.xlabel(feature_name, fontsize=11, labelpad=10, color='#555555')
    plt.ylabel('Frequency', fontsize=11, labelpad=10, color='#555555')

    # Render
    plt.tight_layout()
    plt.show()


###############################################################################
########################### Imputer functions #################################
###############################################################################

def impute_iterative(df: pd.DataFrame, 
                     target_col: str = None,
                     max_iter: int = 10, 
                     random_state: int = 42, 
                     estimator = None) -> pd.DataFrame:
    """
    Fills NaN values using IterativeImputer while ignoring the target column.
    
    Parameters:
    - df: Input pandas DataFrame
    - target_col: Name of the target feature to exclude from imputation
    - max_iter: Maximum number of imputation rounds
    - random_state: Seed for reproducibility
    - estimator: The machine learning model used to predict missing values 
    """
    df_imputed = df.copy()
    
    # Isolate and remove the target column if provided
    target_series = None
    if target_col and target_col in df_imputed.columns:
        target_series = df_imputed.pop(target_col)
    
    num_cols = df_imputed.select_dtypes(include=['number']).columns
    cat_cols = df_imputed.select_dtypes(exclude=['number']).columns
    
    if len(num_cols) > 0:
        imputer = IterativeImputer(estimator=estimator, 
                                   max_iter=max_iter, 
                                   random_state=random_state)
        df_imputed[num_cols] = imputer.fit_transform(df_imputed[num_cols])
        
    # Check if categorical columns still have missing data
    if len(cat_cols) > 0 and df_imputed[cat_cols].isna().any().any():
        print(f"Warning: Categorical columns {list(cat_cols)} still contain NaNs. "
              "Encode them to numeric before using IterativeImputer to fill them.")
              
    # Reattach the target column unmodified
    if target_series is not None:
        df_imputed[target_col] = target_series
        
    return df_imputed

def impute_simple(df: pd.DataFrame, 
                  target_col: str = None,
                  num_strategy: str = 'mean', 
                  cat_strategy: str = 'most_frequent', 
                  cat_fill_value: str = 'Missing') -> pd.DataFrame:
    """
    Fills NaN values using SimpleImputer while ignoring the target column.
    
    Parameters:
    - df: Input pandas DataFrame
    - target_col: Name of the target feature to exclude from imputation
    - num_strategy: Strategy for numeric columns ('mean', 'median', 'most_frequent', 'constant')
    - cat_strategy: Strategy for categorical columns ('most_frequent', 'constant')
    - cat_fill_value: The filler value if cat_strategy='constant'
    """
    df_imputed = df.copy()
    
    # Isolate and remove the target column if provided
    target_series = None
    if target_col and target_col in df_imputed.columns:
        target_series = df_imputed.pop(target_col)
    
    # Identify numeric and non-numeric (categorical) columns
    num_cols = df_imputed.select_dtypes(include=['number']).columns
    cat_cols = df_imputed.select_dtypes(exclude=['number']).columns
    
    # Impute numeric columns
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy=num_strategy)
        df_imputed[num_cols] = num_imputer.fit_transform(df_imputed[num_cols])
        
    # Impute categorical columns
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value=cat_fill_value)
        df_imputed[cat_cols] = cat_imputer.fit_transform(df_imputed[cat_cols])
        
    # Reattach the target column unmodified
    if target_series is not None:
        df_imputed[target_col] = target_series
        
    return df_imputed
