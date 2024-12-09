# # Import necessary libraries
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import joblib

# # Step 1: Load the CSV data
# # Assuming your CSV file has columns: 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'
data = pd.read_csv('Fertilizer.csv')

['Urea' 'DAP' 'Fourteen-Thirty Five-Fourteen' 'Twenty Eight-Twenty Eight'
 'Seventeen-Seventeen-Seventeen' 'Twenty-Twenty'
 'Ten-Twenty Six-Twenty Six']
# print(data['Fertilizer Name'].unique())
# # Step 2: Preprocess data
# # Extract features (X) and labels (y)
# X = data[["Nitrogen","Potassium","Phosphorous"]]  # Features
# y = data['Fertilizer Name']  # Label (crop type)

# # Step 3: Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Initialize and train the Random Forest model
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Step 5: Evaluate the model
# y_pred = rf_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")

# # Step 6: Save the trained model using joblib
# joblib.dump(rf_model, 'random_forest_model_ferlilizer.pkl')
# print("Model saved successfully!")

# # Step 7: Load the saved model using joblib
# loaded_model = joblib.load('random_forest_model_ferlilizer.pkl')
# print("Model loaded successfully!")

# # Step 8: Predict using the loaded model
# # Let's say we want to predict the crop label for a new sample
# new_data_df = pd.DataFrame([[50, 30, 60]], columns=["Nitrogen","Potassium","Phosphorous"])  # Example new data (N, P, K, temperature, humidity, ph, rainfall)
# prediction = loaded_model.predict(new_data_df)
# print(f"Predicted label for the new data: {prediction[0]}")
