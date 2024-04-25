import tkinter as tk
from tkinter import ttk, filedialog
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/futurexskill/projects/main/knn-classification/purchase_history.csv")

# Preprocess the data
gender_encoded = pd.get_dummies(df['Gender'], drop_first=True)
df = pd.concat([df, gender_encoded], axis=1)
x = df[['Male', 'Age', 'Salary', 'Price']].to_numpy()
y = df['Purchased'].to_numpy()

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the model
k = 5
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(x_train, y_train)

# Save the trained model and scaler
with open('knn_model.pickle', 'wb') as f:
    pickle.dump(knn_model, f)

with open('scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)

# Function to make predictions
def predict_purchase(gender, age, salary, price):
    # Load the trained model and scaler
    with open('knn_model.pickle', 'rb') as f:
        knn_model = pickle.load(f)

    with open('scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)

    # Encode gender
    gender_encoded = 1 if gender == 'Male' else 0
    
    # Scale the input features
    input_features = scaler.transform([[gender_encoded, age, salary, price]])
    
    # Make prediction
    prediction = knn_model.predict(input_features)[0]
    
    return prediction

# Function to get accuracy score
def get_accuracy_score():
    y_pred = knn_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return f"Customer behaviour model with {accuracy*100:.0f}% accuracy"


# Function to predict purchases for customers in CSV file
def predict_from_csv(file_path):
    df = pd.read_csv(file_path)
    predictions = []
    for index, row in df.iterrows():
        gender = row['Gender']
        age = row['Age']
        salary = row['Salary']
        price = row['Price']
        prediction = predict_purchase(gender, age, salary, price)
        predictions.append(prediction)
    return predictions

# Create the main window
root = tk.Tk()
root.title("Purchase Prediction")

# Customize the UI colors and fonts
root.configure(bg="#55aabb")  # Set background color
root.geometry("1080x720")       # Set window size

# Accuracy Score Label
accuracy_score_label = ttk.Label(root, text=f'{get_accuracy_score()}', font=("Montserrat", 32, "bold"), background = "#55aabb")
accuracy_score_label.pack(padx=20, pady=20)


# Create a frame for the content
frame = ttk.Frame(root, padding="50")
frame.pack()

# Gender Label and Combobox
gender_label = ttk.Label(frame, text="Gender:", font=("Montserrat", 20))
gender_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

gender_combobox = ttk.Combobox(frame, values=["Male", "Female"], font=("Montserrat", 20))
gender_combobox.grid(row=0, column=1, padx=10, pady=10)
gender_combobox.current(0)

# Age Label and Entry
age_label = ttk.Label(frame, text="Age:", font=("Montserrat", 20))
age_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

age_entry = ttk.Entry(frame, font=("Montserrat", 20))
age_entry.grid(row=1, column=1, padx=10, pady=10)

# Salary Label and Entry
salary_label = ttk.Label(frame, text="Salary:", font=("Montserrat", 20))
salary_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")

salary_entry = ttk.Entry(frame, font=("Montserrat", 20))
salary_entry.grid(row=2, column=1, padx=10, pady=10)

# Price Label and Entry
price_label = ttk.Label(frame, text="Price:", font=("Montserrat", 20))
price_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")

price_entry = ttk.Entry(frame, font=("Montserrat", 20))
price_entry.grid(row=3, column=1, padx=10, pady=10)

# Predict Button for single customer prediction
def predict_purchase_and_display():
    gender = gender_combobox.get()
    age = float(age_entry.get())
    salary = float(salary_entry.get())
    price = float(price_entry.get())
    prediction = predict_purchase(gender, age, salary, price)
    prediction_text = "Likely to purchase" if prediction == 1 else "Not likely to purchase"
    prediction_label.config(text=f'Predicted Purchase: {prediction_text}')

predict_button = ttk.Button(frame, text="Predict ", command=predict_purchase_and_display)
predict_button.grid(row=4, column=0, padx=30, pady=30)

# OR Label
or_label = ttk.Label(frame, text="OR", font=("Montserrat", 20))
or_label.grid(row=4, column=1, padx=20, pady=20)

# Upload CSV Button
def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        predictions = predict_from_csv(file_path)
        prediction_text = "\n".join([f"Customer {i+1}: {'Likely to purchase' if pred == 1 else 'Not likely to purchase'}" for i, pred in enumerate(predictions)])
        prediction_label.config(text=prediction_text)

upload_button = ttk.Button(frame, text="Upload CSV", command=upload_csv)
upload_button.grid(row=4, column=2, padx=20, pady=20)

# Prediction Label
prediction_label = ttk.Label(frame, text="", font=("Montserrat", 20, "bold"), foreground="#336699")
prediction_label.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()
