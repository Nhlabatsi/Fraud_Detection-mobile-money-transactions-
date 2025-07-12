# Fraud_Detection-mobile-money-transactions-
The Real-Time Fraud Detection System is an AI-powered solution designed to identify fraudulent transactions in real-time using a Graph Neural Network (GNN) model.
The system consists of:

A FastAPI backend for model inference and transaction processing.

A Streamlit dashboard for real-time monitoring and analytics.

Key Features
✅ Real-time fraud detection with configurable risk thresholds
✅ Agent-based monitoring with customizable risk tolerances
✅ Interactive dashboard with live transaction visualization
✅ Fraud analytics with risk score distribution and trends
✅ Model performance tracking (accuracy, latency, F1-score)
✅ Alerting system for high-risk transactions

3. Installation & Setup
3.1 Backend Setup
Clone the repository:

bash
Copy
Download
git clone https://github.com/your-repo/fraud-detection.git
cd fraud-detection/backend
Install dependencies:

bash
Copy
Download
pip install -r requirements.txt
Run the FastAPI server:

bash
Copy
Download
uvicorn api:app --reload --host 0.0.0.0 --port 8000
The API will be available at http://localhost:8000

Swagger docs: http://localhost:8000/docs

3.2 Dashboard Setup
Navigate to the frontend directory:

bash
Copy
Download
cd ../frontend
Install dependencies:

bash
Copy
Download
pip install -r requirements.txt
Run the Streamlit dashboard:

bash
Copy
Download
streamlit run dashboard.py
The dashboard will open at http://localhost:8501

4. Usage Guide
4.1 Backend API Endpoints
Endpoint	Method	Description
/predict	POST	Predict fraud risk for a transaction
/agents/create	POST	Create a new monitoring agent
/agents/list	GET	List all agents
/agents/{agent_id}/transactions	GET	Get agent transaction history
/statistics	GET	Get system-wide fraud statistics
4.2 Dashboard Features
📊 Live Transactions Tab
Displays real-time transactions with risk scores.

Color-coded risk levels (Minimal → Critical).

Timeline view of transaction risk trends.

🚨 Fraud Alerts Tab
Lists high-risk transactions flagged by the model.

Shows transaction details (amount, location, risk indicators).

📈 Analytics Tab
Risk score distribution (histogram).

Fraud by agent & time of day (bar charts).

Amount vs. risk score (scatter plot).

🤖 Model Performance Tab
Accuracy, Precision, Recall, F1 trends.

Latency monitoring (ms per prediction).
🏗️ Architecture
flowchart TB
    Frontend -->|API Calls| Backend
    Backend -->|Model Inference| GNN
    Backend -->|Logs| Database[(PostgreSQL)]
    
 Graph Neural Network (GNN) fraud detection model:

Graph Attention Network (GAT) Fraud Detection Model
Overview
This model implements a Graph Attention Network (GAT) for detecting fraudulent transactions in financial networks. The architecture leverages graph-structured data where transactions are represented as nodes and relationships between them as edges.

Key Features
Graph Attention Network implementation using Spektral's GATConv

Scalable training with sparse adjacency matrices

Production-ready serving with TensorFlow SavedModel

Feature normalization integrated into the serving pipeline

Flexible input handling for both sparse and dense adjacency matrices

Model Architecture
Components
1. GATConvWrapper
python
Copy
Download
class GATConvWrapper(tf.keras.layers.Layer)
Wraps Spektral's GATConv layer for TensorFlow 2.x compatibility

Handles attention heads configuration (multi-head attention)

Parameters:

units: Output dimension

attn_heads: Number of attention heads

concat_heads: Whether to concatenate head outputs

2. GNNClassifier
python
Copy
Download
class GNNClassifier(tf.keras.Model)
Two-layer GAT architecture with dropout regularization

Architecture:

First GAT layer (4 attention heads, ReLU activation)

Dropout (30%)

Second GAT layer (single attention head)

Dropout (30%)

Dense classification layer

3. GNNExportWrapper
python
Copy
Download
class GNNExportWrapper(tf.Module)
Production serving wrapper

Includes built-in feature normalization

Provides two serving functions:

serve_sparse: For sparse adjacency matrices

serve_dense: For dense adjacency matrices

Data Preparation
Input Format
The model expects graph data in a tuple format with:

x: Node features (transactions) as numpy array

y: Labels as numpy array

a: Adjacency matrix (sparse or dense)

Preprocessing
Feature Scaling: StandardScaler normalization

Adjacency Matrix Conversion: Handles both sparse and dense formats

Train-Test Split: Stratified 80-20 split

Training Pipeline
Key Steps
Data Loading: Load and preprocess transaction data

Model Initialization: Build GNN architecture

Training Loop:

Uses Adam optimizer (learning rate=0.01)

Sparse Categorical Crossentropy loss

Batch training on full graph

Evaluation:

Classification report

Probability threshold at 0.4 for fraud detection

Model Export:

Saves as TensorFlow SavedModel

Includes scaler parameters
