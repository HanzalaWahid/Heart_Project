import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

def load_data(file_path = 'data/heart.csv' , target_column = 'target' ):
    df = pd.read_csv(file_path)
    return df

def preprocess_data (df , target_column = 'target' , scale = True):
    X = df.drop(target_column, axis=1)
    Y = df[target_column]

    # Handle class imbalance with stratify
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y)

    # Feature scaling
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test, scaler
    
    