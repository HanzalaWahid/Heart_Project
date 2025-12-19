import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

def load_data(file_path = 'data/heart.csv' , target_column = 'target' ):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_column="target", scale=True):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X.columns,
            index=X_test.index
        )

    return X_train, X_test, y_train, y_test, scaler
