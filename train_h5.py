import os
import sys
import subprocess

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
except ImportError as e:
    print(f"Missing dependency: {e}. Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn", "joblib", "tensorflow"])
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

def main():
    print("Loading dataset...")
    df = pd.read_csv('media/heart-disease-dataset.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Building Keras model...")
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(X_test_scaled, y_test))

    print("Evaluating...")
    loss, acc = model.evaluate(X_test_scaled, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    # Save to standard h5 format
    model.save('best_model.h5')
    joblib.dump(scaler, 'scaler_h5.pkl')
    print("Saved best_model.h5 and scaler_h5.pkl gracefully!")

if __name__ == "__main__":
    main()
