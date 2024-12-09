import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# Load the dataset
file_path = 'C:\\Users\\hp\\Desktop\\BitCoin Price Prediction\\Bitcoin Dataset.csv'  # Update this path accordingly
df = pd.read_csv(file_path)

# Handle missing values by dropping rows with any NaNs
df.dropna(inplace=True)

# Prepare the features and target variable
X = df[['open', 'high', 'low', 'volume']]
y = df['close']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to make predictions
def predict_price(open_price, high_price, low_price, volume):
    input_data = np.array([[open_price, high_price, low_price, volume]])
    predicted_price = model.predict(input_data)
    return predicted_price[0]

def show_result():
    try:
        open_price = float(entry_open.get())
        high_price = float(entry_high.get())
        low_price = float(entry_low.get())
        volume = float(entry_volume.get())
        month = int(entry_month.get())
        year = int(entry_year.get())

        predicted_close_price = predict_price(open_price, high_price, low_price, volume)
        messagebox.showinfo("Result", f"The predicted closing price for Bitcoin in {month}/{year} is: {predicted_close_price:.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for price and volume")

def show_data():
    data = df.head(20).to_string()
    messagebox.showinfo("First 20 Lines of Data", data)

def show_accuracy():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    messagebox.showinfo("Model Evaluation", f"Mean Squared Error: {mse:.2f}")

def show_graph():
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.show()

def toggle_fullscreen(event=None):
    global fullscreen
    fullscreen = not fullscreen
    root.attributes("-fullscreen", fullscreen)

def end_fullscreen(event=None):
    global fullscreen
    fullscreen = False
    root.attributes("-fullscreen", False)

# Create the main window
root = tk.Tk()
root.title("BITCOIN MARKET ANALYSIS AND PREDICTION")

# Set background image
bg_image_path = 'C:\\Users\\hp\\Desktop\\BitCoin Price Prediction\\Bitcoin_Image.jpg'
bg_image = Image.open(bg_image_path)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create labels and entries
frame = tk.Frame(root, bg="black")
frame.place(relx=0.5, rely=0.5, anchor="center")

tk.Label(frame, text="Month (1-12):", fg="white", bg="black").grid(row=0, column=0, padx=10, pady=5)
entry_month = tk.Entry(frame)
entry_month.grid(row=0, column=1, padx=10, pady=5)

tk.Label(frame, text="Year (e.g., 2024):", fg="white", bg="black").grid(row=1, column=0, padx=10, pady=5)
entry_year = tk.Entry(frame)
entry_year.grid(row=1, column=1, padx=10, pady=5)

tk.Label(frame, text="Opening Price:", fg="white", bg="black").grid(row=2, column=0, padx=10, pady=5)
entry_open = tk.Entry(frame)
entry_open.grid(row=2, column=1, padx=10, pady=5)

tk.Label(frame, text="Highest Price:", fg="white", bg="black").grid(row=3, column=0, padx=10, pady=5)
entry_high = tk.Entry(frame)
entry_high.grid(row=3, column=1, padx=10, pady=5)

tk.Label(frame, text="Lowest Price:", fg="white", bg="black").grid(row=4, column=0, padx=10, pady=5)
entry_low = tk.Entry(frame)
entry_low.grid(row=4, column=1, padx=10, pady=5)

tk.Label(frame, text="Volume:", fg="white", bg="black").grid(row=5, column=0, padx=10, pady=5)
entry_volume = tk.Entry(frame)
entry_volume.grid(row=5, column=1, padx=10, pady=5)

# Create buttons
button_result = tk.Button(frame, text="Show Result", command=show_result, bg="green", fg="white")
button_result.grid(row=6, column=0, padx=10, pady=5)

button_data = tk.Button(frame, text="Show Data", command=show_data, bg="blue", fg="white")
button_data.grid(row=6, column=1, padx=10, pady=5)

button_accuracy = tk.Button(frame, text="Show Accuracy and Precision", command=show_accuracy, bg="purple", fg="white")
button_accuracy.grid(row=7, column=0, padx=10, pady=5)

button_graph = tk.Button(frame, text="Show Graph", command=show_graph, bg="orange", fg="white")
button_graph.grid(row=7, column=1, padx=10, pady=5)

# Bind ESC and F11 keys
root.bind("<Escape>", end_fullscreen)
root.bind("<F11>", toggle_fullscreen)

# Start in fullscreen mode
fullscreen = True
root.attributes("-fullscreen", fullscreen)

root.mainloop()
