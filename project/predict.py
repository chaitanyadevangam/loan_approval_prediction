import pickle
import numpy as np
import joblib

# Load the trained model using joblib
loaded_model = joblib.load("C:/Users/chait/Downloads/project/train_model (1).joblib")
input_data = np.array([0,1,1,4100000,12200000,8, 417,2700000,2200000,880000,3300000,26])


# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data.reshape(1, -1)

# Make prediction using the loaded model
prediction = loaded_model.predict(input_data_reshaped)

# Print the prediction result
print(f"Prediction: {prediction}")

if prediction[0] == 0:
    print('The loan is not approved')
else:
    print('The loan is approved')



