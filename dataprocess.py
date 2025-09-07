# =================================================
# Data Loading Module
# =================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import Config
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.num_features = Config.NUM_FEATURES
        self.cat_features = Config.CAT_FEATURES
    
    def load_data(self):
        """Load data from all sheets"""
        sheet_names = pd.ExcelFile(self.file_path).sheet_names
        print(f"Excel file contains the following sheets: {sheet_names}")
        
        dfs = []
        for sheet_name in sheet_names:
            df_sheet = pd.read_excel(self.file_path, sheet_name=sheet_name, skiprows=1)
            dfs.append(df_sheet)
            print(f"Sheet '{sheet_name}' contains {len(df_sheet)} rows of data")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"After merging, there are {len(df)} rows of data in total")
        return df
    def handle_imbalanced_data(self, X, y, method='smote'):
       
        if method == 'smote':
            smote = SMOTE(random_state=Config.RANDOM_SEED)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=Config.RANDOM_SEED)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
        elif method == 'RandomOverSampler':
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=Config.RANDOM_SEED)
            X_resampled, y_resampled = ros.fit_resample(X, y)
        elif method  == 'SMOTETomek':
            from imblearn.combine import SMOTETomek
            smote_tomek = SMOTETomek(random_state=Config.RANDOM_SEED)
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        elif method == 'SMOTEENN':
            from imblearn.combine import SMOTEENN
            smote_enn = SMOTEENN(random_state=Config.RANDOM_SEED)
            X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        elif method == 'BorderlineSMOTE':
            from imblearn.over_sampling import BorderlineSMOTE
            smote_borderline1 = BorderlineSMOTE(kind='borderline-1', random_state=Config.RANDOM_SEED)
            X_resampled, y_resampled = smote_borderline1.fit_resample(X, y)

        else:
            X_resampled, y_resampled = X, y
    
        return X_resampled, y_resampled
    def preprocess_data(self, df):
        """Data preprocessing"""
        # Handle missing values
        print(f"\nData shape: {df.shape}")
        print(f"Missing value statistics:")
        print(df.isnull().sum())
        
        for feature in self.num_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0)
        
        for feature in self.cat_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna("Unknown")
        
        # Process label column
        if 'score' in df.columns:
            df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        else:
            print("Warning: 'score' column not found, will create default labels")
            df['score'] = 0
        
        # Check label distribution
        unique_labels, counts = np.unique(df['score'], return_counts=True)
        print(f"\nLabel distribution: {dict(zip(unique_labels, counts))}")
        
        # Preprocessing pipeline
        numeric_pipe = Pipeline([('scaler', StandardScaler())])
        categorical_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipe, self.num_features),
                ('cat', categorical_pipe, self.cat_features)
            ])
        
        X_processed = preprocessor.fit_transform(df)
        print(f"Preprocessed feature dimensions: {X_processed.shape}")
        
        return X_processed, df['score'].values
    
    def create_simple_classification_dataset_with_split(self, df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """Create dataset for traditional machine learning classification, maintaining the same split as GNN"""
        from config import Config
        import random
        
        # Handle missing values
        print(f"\nData shape: {df.shape}")
        print(f"Missing value statistics:")
        print(df.isnull().sum())
        
        for feature in self.num_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0)
        
        for feature in self.cat_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna("Unknown")
        
        # Process label column
        if 'score' in df.columns:
            df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        else:
            print("Warning: 'score' column not found, will create default labels")
            df['score'] = 0
        
        # Check label distribution
        unique_labels, counts = np.unique(df['score'], return_counts=True)
        print(f"\nLabel distribution: {dict(zip(unique_labels, counts))}")
        
        # Preprocessing pipeline
        numeric_pipe = Pipeline([('scaler', StandardScaler())])
        categorical_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipe, self.num_features),
                ('cat', categorical_pipe, self.cat_features)
            ])
        
        X_processed = preprocessor.fit_transform(df)
        y = df['score'].values
        print(f"Preprocessed feature dimensions: {X_processed.shape}")
        
        # Use the same random seed and splitting strategy as GNN
        np.random.seed(Config.RANDOM_SEED)
        random.seed(Config.RANDOM_SEED)
        
        # Get indices and shuffle
        n_samples = X_processed.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Stratified sampling by class to ensure consistent data distribution
        unique_classes = np.unique(y)
        train_indices, val_indices, test_indices = [], [], []
        
        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            np.random.shuffle(class_indices)
            
            n_train = int(len(class_indices) * train_ratio)
            n_val = int(len(class_indices) * val_ratio)
            
            train_indices.extend(class_indices[:n_train])
            val_indices.extend(class_indices[n_train:n_train+n_val])
            test_indices.extend(class_indices[n_train+n_val:])
        
        # Create split datasets
        X_train = X_processed[train_indices]
        X_val = X_processed[val_indices]
        X_test = X_processed[test_indices]
        
        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]
        
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Val samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }