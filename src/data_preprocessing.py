
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.medians = {}
        self.numerical_cols = [
            'tenure',
            'MonthlyCharges',
            'TotalCharges',
            'avg_monthly_per_tenure'
        ]

    # 1. LOAD DATA
    def load_data(self, filepath):

        print(f"Loading data from {filepath}...")

        df = pd.read_csv(filepath)

        print(f"Dataset loaded successfully: {df.shape}")

        return df

    # 2. HANDLE MISSING VALUES
    def handle_missing_values(self, X_train, X_test):

        print("\nHandling missing values...")

        numeric_cols = [
            'tenure',
            'MonthlyCharges',
            'TotalCharges'
        ]

        for col in numeric_cols:

            if col in X_train.columns:

                # Convert values into numeric
                X_train[col] = pd.to_numeric(
                    X_train[col].astype(str).str.strip(),
                    errors='coerce'
                )

                X_test[col] = pd.to_numeric(
                    X_test[col].astype(str).str.strip(),
                    errors='coerce'
                )

                # Calculate median using training data only
                median_value = X_train[col].median()

                if pd.isna(median_value):
                    median_value = 0

                self.medians[col] = median_value

                # Fill missing values
                X_train[col] = X_train[col].fillna(median_value)
                X_test[col] = X_test[col].fillna(median_value)

                print(
                    f"   {col}: filled missing values "
                    f"with median {median_value:.2f}"
                )

        print("Missing value handling completed.")

        return X_train, X_test

    # 3. FEATURE ENGINEERING
    def feature_engineering(self, df):

        print("\nPerforming feature engineering...")

        df = df.copy()

        # Tenure groups
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[-np.inf, 12, 24, 48, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # Average monthly charge per tenure month
        df['avg_monthly_per_tenure'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']
        )

        # Count additional services
        service_cols = [
            'PhoneService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport'
        ]

        df['num_services'] = 0

        for col in service_cols:

            if col in df.columns:

                df['num_services'] += (
                    df[col].astype(str).str.strip() == 'Yes'
                ).astype(int)

        # Count internet service separately
        if 'InternetService' in df.columns:

            df['num_services'] += (
                df['InternetService'].astype(str).str.strip() != 'No'
            ).astype(int)

        print("Created 3 new features:")
        print(f"   tenure_group: {df['tenure_group'].unique()}")
        print(
            f"   avg_monthly_per_tenure: "
            f"{df['avg_monthly_per_tenure'].min():.2f} - "
            f"{df['avg_monthly_per_tenure'].max():.2f}"
        )
        print(
            f"   num_services: "
            f"{df['num_services'].min()} - "
            f"{df['num_services'].max()}"
        )

        return df

    # 4. ENCODE CATEGORICAL FEATURES
    def encode_features(self, df, is_training=False):

        df = df.copy()

        # Drop customer ID
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])

        # Binary encoding
        binary_cols = [
            'Partner',
            'Dependents',
            'PhoneService',
            'PaperlessBilling',
            'gender'
        ]

        for col in binary_cols:

            if col not in df.columns:
                continue

            values = df[col].astype(str).str.strip()

            if col == 'gender':
                mapping = {
                    'Female': 0,
                    'Male': 1
                }
            else:
                mapping = {
                    'No': 0,
                    'Yes': 1
                }

            df[col] = values.map(mapping).fillna(-1).astype(int)

        # Encode remaining categorical columns
        categorical_cols = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        for col in categorical_cols:

            values = df[col].fillna('Unknown').astype(str).str.strip()

            if is_training:

                unique_values = sorted(values.unique())

                self.label_encoders[col] = {
                    value: index
                    for index, value in enumerate(unique_values)
                }

                print(
                    f"   Encoded {col}: "
                    f"{len(unique_values)} categories"
                )

            mapping = self.label_encoders.get(col, {})

            # Unknown categories receive -1
            df[col] = values.map(mapping).fillna(-1).astype(int)

        print(
            f"Categorical encoding completed: "
            f"{len(binary_cols) + len(categorical_cols)} columns checked."
        )

        return df

    # 5. SCALE NUMERICAL FEATURES
    def scale_features(
        self,
        X_train,
        X_test=None,
        is_training=True
    ):

        X_train = X_train.copy()

        if X_test is not None:
            X_test = X_test.copy()

        numerical_cols = [
            col for col in self.numerical_cols
            if col in X_train.columns
        ]

        if is_training:

            X_train[numerical_cols] = self.scaler.fit_transform(
                X_train[numerical_cols]
            )

            if X_test is not None:

                X_test[numerical_cols] = self.scaler.transform(
                    X_test[numerical_cols]
                )

        else:

            X_train[numerical_cols] = self.scaler.transform(
                X_train[numerical_cols]
            )

        print("\nNumerical features scaled successfully.")

        return X_train, X_test

    # 6. COMPLETE PREPROCESSING PIPELINE
    def prepare_data(
        self,
        df,
        test_size=0.2,
        random_state=42
    ):

        print("\n" + "=" * 60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 60)

        df = df.copy()

        # Check target column
        if 'Churn' not in df.columns:
            raise ValueError("Churn column not found in dataset.")

        # Convert target into binary values
        y = (
            df['Churn']
            .astype(str)
            .str.strip()
            .map({'Yes': 1, 'No': 0})
        )

        if y.isnull().any():
            raise ValueError(
                "Churn contains missing or unexpected values."
            )

        y = y.astype(int)

        # Separate features and target
        X = df.drop(columns=['Churn'])

        # Remove customer ID
        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        X_train = X_train.copy()
        X_test = X_test.copy()

        print("\nTrain-Test Split:")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")

        # Handle missing values using training medians
        X_train, X_test = self.handle_missing_values(
            X_train,
            X_test
        )

        # Feature engineering
        X_train = self.feature_engineering(X_train)
        X_test = self.feature_engineering(X_test)

        # Encode categorical features
        X_train = self.encode_features(
            X_train,
            is_training=True
        )

        X_test = self.encode_features(
            X_test,
            is_training=False
        )

        # Ensure identical column order
        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0
        )

        # Scale numerical features
        X_train, X_test = self.scale_features(
            X_train,
            X_test,
            is_training=True
        )

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        return X_train, X_test, y_train, y_test



if __name__ == "__main__":

    preprocessor = DataPreprocessor()

    # Get the main project directory (one level above src)
    project_dir = Path(__file__).resolve().parent.parent

    # Correct dataset location
    filepath = project_dir / "data" / "telco_comm_churn.csv"

    print("Dataset path:", filepath)
    print("Dataset exists:", filepath.exists())

    # Load dataset
    df = preprocessor.load_data(filepath)

    # Run preprocessing
    X_train, X_test, y_train, y_test = (
        preprocessor.prepare_data(df)
    )

    print("\nPREPROCESSING VERIFICATION")
    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)
    print("Training target:", y_train.shape)
    print("Testing target:", y_test.shape)

    print("\nMissing values:", X_train.isnull().sum().sum())
    print("\nALL CHECKS COMPLETED!")
