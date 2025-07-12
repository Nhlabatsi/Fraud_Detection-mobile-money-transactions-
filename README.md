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

