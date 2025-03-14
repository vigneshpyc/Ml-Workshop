# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

# Step 1: Create a simple dataset
data = {
    'Age': ['Young', 'Young', 'Middle-aged', 'Senior', 'Senior', 'Senior', 'Middle-aged', 
            'Young', 'Young', 'Senior', 'Young', 'Middle-aged', 'Middle-aged', 'Senior'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 
               'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 
                'No', 'Yes', 'No'],
    'Buys_Product': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 
                     'Yes', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Encode categorical variables
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store encoder for later decoding

# Step 3: Split features and target
X = df.drop(columns=['Buys_Product'])
y = df['Buys_Product']

# Step 4: Train Decision Tree model
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X, y)

# Step 5: Get runtime input from the user
def get_user_input():
    print("\nEnter your details for prediction:")
    age_input = input("Enter Age (Young/Middle-aged/Senior): ").strip()
    income_input = input("Enter Income (High/Medium/Low): ").strip()
    student_input = input("Are you a Student? (Yes/No): ").strip()
    
    # Convert user input into numerical format using label encoders
    try:
        age_encoded = label_encoders['Age'].transform([age_input])[0]
        income_encoded = label_encoders['Income'].transform([income_input])[0]
        student_encoded = label_encoders['Student'].transform([student_input])[0]
    except ValueError:
        print("\nInvalid input! Please enter correct values.")
        return None  # Return None if input is invalid

    return np.array([[age_encoded, income_encoded, student_encoded]])

# Step 6: Make prediction
user_data = get_user_input()
if user_data is not None:
    prediction = clf.predict(user_data)
    decoded_prediction = label_encoders['Buys_Product'].inverse_transform(prediction)
    print("\n📢 Prediction:", decoded_prediction[0])

# Step 7: Visualize the Decision Tree
plt.figure(figsize=(12, 6))
tree.plot_tree(clf, feature_names=['Age', 'Income', 'Student'], class_names=['No', 'Yes'], filled=True)
plt.show()
