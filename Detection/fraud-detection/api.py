from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
from datetime import datetime
import numpy as np
import json
import os
import asyncio
import httpx
import logging
import sys
import warnings
import tensorflow as tf
from scipy import sparse as sp
import random
import hashlib
from faker import Faker

# Initialize Faker
fake = Faker()

# Add the models directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="API for real-time fraud detection using Graph Neural Networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants - Updated to match scaler_params.json
EXPECTED_FEATURES = 19
FEATURE_NAMES = [
    "amount", "distance", "agent_longitude", "transaction_lat", "transaction_lon",
    "hour", "risk_score", "gender_0", "gender_1",
    "day_of_week_0", "day_of_week_1", "day_of_week_2", "day_of_week_3",
    "day_of_week_4", "day_of_week_5", "day_of_week_6",
    "multi_subscriber_fraud", "is_weekend", "is_business_hours"
]

# Pydantic models
class TransactionFeatures(BaseModel):
    features: List[float]
    neighbors: Optional[List[List[float]]] = []
    adj_size: Optional[int] = 1
    threshold: Optional[float] = 0.4

class PredictionResponse(BaseModel):
    prediction: bool
    fraud_score: float
    confidence: float
    probabilities: List[float]
    threshold: float
    processing_time_ms: float
    timestamp: str
    feature_names: List[str] = FEATURE_NAMES
    expected_features: int = EXPECTED_FEATURES

class AlertConfig(BaseModel):
    email: Optional[str] = None
    webhook_url: Optional[str] = None
    threshold: float = 0.7

class AgentCreateRequest(BaseModel):
    name: str
    type: str
    risk_tolerance: float

class AgentInfo(BaseModel):
    id: str
    name: str
    type: str
    risk_tolerance: float
    created_at: datetime
    total_transactions: int
    fraud_detected: int
    accuracy: float
    avg_processing_time: float
    status: str

class TransactionRecord(BaseModel):
    Transaction_ID: str
    Subscriber_ID: str
    Agent_ID: str
    Timestamp: datetime
    Amount: float
    Distance: float
    Multi_Subscriber_Fraud: bool
    Subscriber_Name: str
    Gender: str
    SIM_Card_ID: str
    Agent_Longitude: float
    Transaction_Lat: float
    Transaction_Lon: float
    Money_Fraud_Label_agent: int
    Date: str
    Hour: int
    DayOfWeek: str
    IsWeekend: bool
    IsBusinessHours: bool
    Risk_Score: float
    National_ID: str
    Fraud_Label: int
    Fraud_Prediction: Optional[bool] = None
    Fraud_Score: Optional[float] = None
    Confidence: Optional[float] = None

# Global variables
model = None
scaler_mean = None
scaler_scale = None
alert_config = AlertConfig()
agents = {}
transaction_history = {}
alert_history = {}

def load_fraud_model():
    global model, scaler_mean, scaler_scale
    try:
        model_path = os.path.join(os.path.dirname(__file__), "models", "gnn_fraud_classifier")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        model = tf.saved_model.load(model_path)
        logger.info("Model loaded from %s", model_path)

        # Load scaler parameters from the provided JSON
        scaler_params = {
            "mean_": [5.390538717620075e-10, -2.7264468371868134e-10, 2.6906491257250307e-10, 
                     6.628688424825668e-10, 1.0943040251731872e-10, 2.5816261768341063e-09, 
                     1.0728836059570312e-09, 0.462, 0.538, 0.142, 0.143, 0.142, 0.144, 
                     0.144, 0.141, 0.144, 0.02, 0.286, 0.404],
            "scale_": [1.0000000009951178, 0.9999999996131818, 1.0000000001110667, 
                      1.000000000011781, 1.000000000932205, 0.9999999888838185, 
                      1.0000000043572561, 0.4985539088203035, 0.4985539088203034, 
                      0.34905013966477366, 0.35007284956134377, 0.3490501396647735, 
                      0.35108973211986677, 0.35108973211986677, 0.34802155105682614, 
                      0.35108973211986677, 0.1399999999999995, 0.4518893669915207, 
                      0.4906974628016783]
        }
        
        scaler_mean = np.array(scaler_params['mean_'], dtype=np.float32)
        scaler_scale = np.array(scaler_params['scale_'], dtype=np.float32)
        
        # Verify feature count matches
        if len(scaler_mean) != EXPECTED_FEATURES or len(scaler_scale) != EXPECTED_FEATURES:
            raise ValueError(f"Scaler parameters have incorrect length. Expected {EXPECTED_FEATURES} features")
            
        logger.info("Scaler loaded successfully with %d features", EXPECTED_FEATURES)

    except Exception as e:
        logger.error("Error loading model: %s", str(e))
        raise

def validate_features(features: List[float]) -> np.ndarray:
    if len(features) != EXPECTED_FEATURES:
        raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {len(features)}")
    
    features_array = np.array(features, dtype=np.float32).reshape(1, -1)
    if not np.all(np.isfinite(features_array)):
        raise ValueError("Features contain NaN or infinite values")
    
    return (features_array - scaler_mean) / scaler_scale

def predict_single(features: List[float], neighbors: List[List[float]] = None, threshold: float = 0.4) -> Dict[str, Any]:
    start_time = time.time()
    norm_features = validate_features(features)
    
    adj_matrix = sp.identity(norm_features.shape[0], format='coo', dtype=np.float32)
    indices = np.vstack((adj_matrix.row, adj_matrix.col)).T
    values = adj_matrix.data
    dense_shape = adj_matrix.shape

    adj_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)
    adj_tensor = tf.sparse.reorder(adj_tensor)

    result = model.serve_sparse(
        tf.convert_to_tensor(norm_features, dtype=tf.float32),
        adj_tensor
    )

    probabilities = result['probabilities'].numpy()[0]
    fraud_score = float(probabilities[1])
    prediction = fraud_score > threshold

    return {
        'prediction': prediction,
        'fraud_score': fraud_score,
        'confidence': abs(fraud_score - threshold),
        'probabilities': [float(probabilities[0]), float(probabilities[1])],
        'threshold': threshold,
        'processing_time_ms': (time.time() - start_time) * 1000
    }

async def send_fraud_alert(transaction_data: Dict[str, Any]):
    try:
        if alert_config.webhook_url:
            async with httpx.AsyncClient() as client:
                await client.post(alert_config.webhook_url, json=transaction_data, timeout=10.0)
            logger.info("Webhook alert sent")
        if alert_config.email:
            pass  # Email alert logic here
    except Exception as e:
        logger.error("Alert failed: %s", str(e))

def generate_agent_id() -> str:
    """Generate unique agent ID"""
    return f"AGENT_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8].upper()}"

def generate_sophisticated_transaction(transaction_type: str = "normal") -> tuple:
    """Generate transaction in same format as create_transaction_dataset"""
    np.random.seed(int(time.time() * 1000) % 10000)

    # Base transaction patterns
    if transaction_type == "fraudulent":
        amount = np.random.choice([
            np.random.exponential(500) + 1000,
            np.random.uniform(1, 10),
            np.random.uniform(99, 101)
        ])
        distance = np.random.choice([
            np.random.exponential(200) + 100,
            0.0
        ])
        hour = np.random.choice([2, 3, 4, 23, 0, 1])
        risk_score = np.random.beta(7, 2)  # Higher risk for fraud
        multi_subscriber = True
        money_fraud = 1
    elif transaction_type == "suspicious":
        amount = np.random.exponential(200) + 100
        distance = np.random.exponential(50) + 20
        hour = np.random.choice([22, 23, 0, 1, 6, 7])
        risk_score = np.random.beta(4, 4)  # Moderate risk
        multi_subscriber = bool(np.random.choice([0, 1]))
        money_fraud = 0
    else:
        # Normal transaction adjustments
        amount = np.random.lognormal(3, 1.5)
        distance = np.random.uniform(0, 0.1)  # Typically under 0.1 km
        hour = np.random.randint(8, 22)
        risk_score = np.random.beta(1, 9)  # Typically under 0.1 (90% of values < 0.1)
        multi_subscriber = False
        money_fraud = 0

    # Location
    agent_longitude = np.random.uniform(-167.49, 177.08)
    transaction_lat = np.random.uniform(-85.68, 78.17)
    transaction_lon = np.random.uniform(-167.56, 177.15)

    # Gender
    gender = np.random.choice(['Male', 'Female'])
    first_name = fake.first_name_male() if gender == 'Male' else fake.first_name_female()
    last_name = fake.last_name()
    full_name = f"{first_name} {last_name}"

    # Day and time
    timestamp = datetime.now()
    day_of_week = timestamp.strftime('%A')
    is_weekend = timestamp.weekday() >= 5
    is_business_hours = 9 <= hour <= 17

    # Fake IDs
    subscriber_id = f"SUB{np.random.randint(1, 28):05d}"
    sim_card_id = f"SIM{int(subscriber_id[-5:]):06d}"
    agent_id = f"AGT{np.random.randint(1, 500):04d}"
    transaction_id = f"TXN{random.randint(1_000_000, 9_999_999):08d}"
    national_id = fake.ssn()

    # Final record
    transaction_record = {
        'Transaction_ID': transaction_id,
        'Subscriber_ID': subscriber_id,
        'Agent_ID': agent_id,
        'Timestamp': timestamp,
        'Amount': round(amount, 2),
        'Distance': round(distance, 4),
        'Multi_Subscriber_Fraud': multi_subscriber,
        'Subscriber_Name': full_name,
        'Gender': gender,
        'SIM_Card_ID': sim_card_id,
        'Agent_Longitude': round(agent_longitude, 6),
        'Transaction_Lat': round(transaction_lat, 6),
        'Transaction_Lon': round(transaction_lon, 6),
        'Money_Fraud_Label_agent': money_fraud,
        'Date': timestamp.date(),
        'Hour': hour,
        'DayOfWeek': day_of_week,
        'IsWeekend': is_weekend,
        'IsBusinessHours': is_business_hours,
        'Risk_Score': round(risk_score, 4),
        'National_ID': national_id,
        'Fraud_Label': int(multi_subscriber or distance > 0.1 or money_fraud == 1 or risk_score > 0.1)
    }

    # Convert to feature vector for API
    gender_0 = 1 if gender == 'Male' else 0
    gender_1 = 1 if gender == 'Female' else 0
    
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    day_of_week_idx = day_map[day_of_week]
    day_of_week_features = [0] * 7
    day_of_week_features[day_of_week_idx] = 1
    
    features = [
        amount, distance, agent_longitude, transaction_lat, transaction_lon,
        hour, risk_score, gender_0, gender_1,
        *day_of_week_features,
        int(multi_subscriber), int(is_weekend), int(is_business_hours)
    ]
    
    return ensure_feature_length(features), transaction_record

def ensure_feature_length(features, expected_length=EXPECTED_FEATURES):
    features = [float(x) if x is not None else 0.0 for x in features]
    if len(features) > expected_length:
        features = features[:expected_length]
    elif len(features) < expected_length:
        features += [0.0] * (expected_length - len(features))
    return features

@app.on_event("startup")
async def startup_event():
    try:
        load_fraud_model()
        logger.info("API started with %d expected features", EXPECTED_FEATURES)
    except Exception as e:
        logger.critical("Startup failed: %s", str(e))
        raise

@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running. See /docs for API documentation."}

@app.get("/health")
async def health():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "expected_features": EXPECTED_FEATURES,
        "feature_names": FEATURE_NAMES,
        "scaler_loaded": scaler_mean is not None,
        "alert_config": alert_config.dict()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionFeatures, background_tasks: BackgroundTasks):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = datetime.now()
    try:
        result = predict_single(
            features=transaction.features,
            neighbors=transaction.neighbors,
            threshold=transaction.threshold
        )
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        response = PredictionResponse(
            prediction=result['prediction'],
            fraud_score=result['fraud_score'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            threshold=result['threshold'],
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )

        # Store transaction in global history
        if isinstance(transaction_history, dict):
            # If using agent-based history, skip here
            pass
        else:
            transaction_history.append({
                **response.dict(),
                "features": transaction.features,
                "timestamp": datetime.now()
            })

        if result['fraud_score'] > alert_config.threshold:
            alert_data = {
                **response.dict(),
                "features": transaction.features,
                "feature_names": FEATURE_NAMES,
                "alert_time": datetime.now().isoformat()
            }
            background_tasks.add_task(send_fraud_alert, alert_data)
            logger.warning("Fraud alert triggered")

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/agents/create")
async def create_agent(request: AgentCreateRequest):
    agent_id = generate_agent_id()
    agents[agent_id] = {
        "id": agent_id, 
        "name": request.name,
        "type": request.type,
        "risk_tolerance": request.risk_tolerance,
        "created_at": datetime.now(),
        "total_transactions": 0,
        "fraud_detected": 0,
        "accuracy": 0.0,
        "avg_processing_time": 0.0,
        "status": "ACTIVE"
    }
    transaction_history[agent_id] = []
    alert_history[agent_id] = []
    return {"status": "success", "agent_id": agent_id}

@app.get("/agents/list")
async def list_agents():
    return {"agents": agents}

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agents[agent_id]

@app.get("/agents/{agent_id}/transactions")
async def get_agent_transactions(agent_id: str, limit: int = 100):
    if agent_id not in transaction_history:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"transactions": transaction_history[agent_id][-limit:]}

@app.get("/agents/{agent_id}/alerts")
async def get_agent_alerts(agent_id: str, limit: int = 100):
    if agent_id not in alert_history:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"alerts": alert_history[agent_id][-limit:]}

@app.post("/agents/{agent_id}/process_transaction")
async def process_agent_transaction(agent_id: str, transaction_type: str = "normal"):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    features, transaction_record = generate_sophisticated_transaction(transaction_type)
    result = predict_single(features=features, threshold=agents[agent_id]['risk_tolerance'])
    
    # Update transaction record with prediction results
    transaction_record.update({
        'Fraud_Prediction': result['prediction'],
        'Fraud_Score': result['fraud_score'],
        'Confidence': result['confidence']
    })
    
    # Update agent stats
    agents[agent_id]['total_transactions'] += 1
    if result['prediction']:
        agents[agent_id]['fraud_detected'] += 1
        alert_history[agent_id].append(transaction_record)
    
    # Store transaction
    transaction_history[agent_id].append(transaction_record)
    
    return {
        "status": "success",
        "transaction": transaction_record,
        "prediction": result
    }

@app.get("/statistics")
async def statistics():
    total = sum(len(txs) for txs in transaction_history.values())
    frauds = sum(1 for txs in transaction_history.values() for t in txs if t.get('Fraud_Label'))
    return {
        "total_transactions": total,
        "fraud_count": frauds,
        "fraud_rate": (frauds / total * 100) if total else 0,
        "average_fraud_score": sum(t.get('Risk_Score', 0) for txs in transaction_history.values() for t in txs) / total if total else 0,
        "last_hour": {
            "transactions": sum(1 for txs in transaction_history.values() for t in txs if (datetime.now() - t['Timestamp']).total_seconds() < 3600),
            "fraud_count": sum(1 for txs in transaction_history.values() for t in txs if t.get('Fraud_Label') and (datetime.now() - t['Timestamp']).total_seconds() < 3600)
        }
    }

@app.post("/update_alert_config")
async def update_alert_config(config: AlertConfig):
    global alert_config
    alert_config = config
    logger.info("Alert config updated: %s", config.json())
    return {"status": "success", "config": alert_config.dict()}

@app.get("/feature_schema")
async def get_feature_schema():
    return {
        "expected_features": EXPECTED_FEATURES,
        "feature_names": FEATURE_NAMES,
        "feature_descriptions": {
            "amount": "Transaction amount in USD",
            "distance": "Distance from home location in km",
            "agent_longitude": "Longitude coordinate of the agent",
            "transaction_lat": "Latitude coordinate of transaction",
            "transaction_lon": "Longitude coordinate of transaction",
            "hour": "Hour of day (0-23)",
            "risk_score": "Precomputed risk score (0-1)",
            "gender_0": "Gender male indicator (1 if male)",
            "gender_1": "Gender female indicator (1 if female)",
            "day_of_week_0": "Monday indicator",
            "day_of_week_1": "Tuesday indicator",
            "day_of_week_2": "Wednesday indicator",
            "day_of_week_3": "Thursday indicator",
            "day_of_week_4": "Friday indicator",
            "day_of_week_5": "Saturday indicator",
            "day_of_week_6": "Sunday indicator",
            "multi_subscriber_fraud": "Linked to multiple accounts (0 or 1)",
            "is_weekend": "Weekend flag (0 or 1)",
            "is_business_hours": "Business hours flag (0 or 1)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=True)