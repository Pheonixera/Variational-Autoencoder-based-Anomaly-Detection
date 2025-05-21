import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import os

# Constants
NSL_KDD_TRAIN_PATH = r"D:\IDS\CIC\PreProcessing\KDDTrain+_20Percent.arff"
NSL_KDD_TEST_PATH = r"D:\IDS\CIC\PreProcessing\KDDTest+.arff"
MODEL_PATH = "vae_model.pth"
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
]

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(filepath, encoders=None, scaler=None):
    from scipy.io import arff
    import tempfile

    # Step 1: Clean ARFF content (remove extra quotes or spaces in attribute values)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        if line.strip().lower().startswith("@attribute"):
            # Clean protocol_type, service, flag lines (e.g., remove quotes and extra spaces)
            if '{' in line and '}' in line:
                attr, values = line.split('{', 1)
                values = values.split('}')[0]
                cleaned_values = ','.join([v.strip().replace("'", "") for v in values.split(',')])
                new_line = f"{attr.strip()} {{{cleaned_values}}}\n"
                cleaned_lines.append(new_line)
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    # Step 2: Save to temp file and read
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.arff') as temp_file:
        temp_file.writelines(cleaned_lines)
        temp_filepath = temp_file.name

    # Step 3: Load using scipy after cleaning
    data, meta = arff.loadarff(temp_filepath)
    df = pd.DataFrame(data)

    # Step 4: Decode byte strings and strip spaces
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].map(lambda x: x.decode('utf-8').strip().replace("'", "") if isinstance(x, bytes) else x.strip().replace("'", ""))

    if 'class' in df.columns and 'attack_type' not in df.columns:
        df.rename(columns={'class': 'attack_type'}, inplace=True)

    if 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])

    original_df = df.copy()

    X = df.drop(columns=['attack_type'])
    y = df['attack_type']

    categorical_cols = ['protocol_type', 'service', 'flag']

    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(df[col])
            X[col] = le.transform(X[col])
            encoders[col] = le
    else:
        for col in categorical_cols:
            X[col] = encoders[col].transform(X[col])

    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    y_binary = y.apply(lambda x: 0 if x == 'normal' else 1)

    return X_scaled, y_binary, original_df, encoders, scaler

# 2. VAE Model Definition
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

# 3. Loss Function
def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# 4. Anomaly Detection and Formatted Output
def detect_anomalies(model, data_tensor, original_df, threshold=0.1):
    model.eval()
    with torch.no_grad():
        reconstructions, _, _ = model(data_tensor)
        errors = torch.mean((data_tensor - reconstructions) ** 2, dim=1).cpu().numpy()

    anomalies = errors > threshold
    results = []

    for i in range(len(anomalies)):
        if anomalies[i]:
            original_row = original_df.iloc[i]
            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.5",
                "anomaly_type": "Network Traffic Anomaly",
                "severity": "High" if errors[i] > threshold * 1.5 else "Medium",
                "deviation_score": float(errors[i] * 100),
                "description": f"Detected abnormal network pattern with reconstruction error {errors[i]:.4f}",
                "action_taken": "Logged for review",
                "protocol": original_row.get('protocol_type', 'unknown')
            }
            formatted = "\n".join([f"{key}: {value}" for key, value in result.items()])
            results.append(formatted + "\n")

    return results

# 5. Main Execution

def main():
    if not os.path.exists(MODEL_PATH):
        # Train new model
        X_train, _, _, encoders, scaler = load_and_preprocess_data(NSL_KDD_TRAIN_PATH)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

        model = VAE(X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(X_train_tensor)
            loss = vae_loss_function(recon_batch, X_train_tensor, mu, logvar)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved to", MODEL_PATH)

        with torch.no_grad():
            recon, _, _ = model(X_train_tensor)
            train_errors = torch.mean((X_train_tensor - recon) ** 2, dim=1).numpy()
            threshold = np.percentile(train_errors, 95)
            print(f"Anomaly threshold set at: {threshold:.4f}")

    else:
        # Load model and test data
        X_test, y_test, original_df, encoders, scaler = load_and_preprocess_data(r"D:\IDS\CIC\PreProcessing\KDDTest+.arff")
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        model = VAE(X_test.shape[1])
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Loaded model from", MODEL_PATH)

        with torch.no_grad():
            recon, _, _ = model(X_test_tensor)
            test_errors = torch.mean((X_test_tensor - recon) ** 2, dim=1).numpy()
            threshold = np.percentile(test_errors, 95)
            print(f"Anomaly threshold set at: {threshold:.4f}")

        anomaly_results = detect_anomalies(model, X_test_tensor, original_df, threshold)
        for res in anomaly_results[:5]:
            print(res)

if __name__ == "__main__":
    main()
