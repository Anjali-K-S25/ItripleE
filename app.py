import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and prepare dataset
df = pd.read_csv("login_dataset.csv")  # Ensure this CSV is in the same folder

# Encode categorical features
le_username = LabelEncoder()
le_ip = LabelEncoder()
le_device = LabelEncoder()

df["username"] = le_username.fit_transform(df["username"])
df["ip_address"] = le_ip.fit_transform(df["ip_address"])
df["device_type"] = le_device.fit_transform(df["device_type"])

# Features and labels
X = df.drop("successful", axis=1)
y = df["successful"]

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(le_username, "le_username.pkl")
joblib.dump(le_ip, "le_ip.pkl")
joblib.dump(le_device, "le_device.pkl")

# Prediction function
def predict(username, ip_address, device_type, login_time_hour):
    try:
        username_enc = le_username.transform([username])[0]
        ip_enc = le_ip.transform([ip_address])[0]
        device_enc = le_device.transform([device_type])[0]
    except ValueError:
        return "Unknown user or device - suspicious login."

    X_new = [[username_enc, ip_enc, device_enc, login_time_hour]]
    pred = model.predict(X_new)[0]

    return "Login Successful - Safe" if pred == 1 else "Warning! Suspicious Login Detected"

# Gradio UI
app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Username"),
        gr.Textbox(label="IP Address"),
        gr.Dropdown(["mobile", "desktop", "tablet"], label="Device Type"),
        gr.Slider(0, 23, step=1, label="Login Hour")
    ],
    outputs="text",
    title="ItripleE - Smart Secure Login",
    description="Check if your login is safe or suspicious based on pattern analysis."
)

app.launch()
