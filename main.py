from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import os
from typing import List, Dict, Any

app = FastAPI(title="Lead Scoring ML API", version="1.0")

# Global variables to store model and encoders
model = None
feature_columns = None
label_encoders = {}
target_column = None
X_test_global = None
y_test_global = None

# --- Load and Train Model ---
def load_and_train_model(csv_path='lead_scoring_dataset.csv'):
    """Load dataset and train the Random Forest model"""
    global model, feature_columns, label_encoders, target_column, X_test_global, y_test_global
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset '{csv_path}' not found!")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Auto-detect target column (look for 'won' or similar)
    possible_targets = ['won', 'target', 'converted', 'is_won', 'outcome', 'label']
    target_column = None
    
    for col in possible_targets:
        if col in df.columns:
            target_column = col
            break
    
    # If not found, assume last column is target
    if target_column is None:
        target_column = df.columns[-1]
        print(f"⚠️ Using last column as target: '{target_column}'")
    else:
        print(f"✅ Target column detected: '{target_column}'")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Store feature columns
    feature_columns = X.columns.tolist()
    
    # Identify and encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"📊 Categorical features: {categorical_cols}")
    print(f"🔢 Numerical features: {[col for col in feature_columns if col not in categorical_cols]}")
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded '{col}' with {len(le.classes_)} unique values")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_test_global = X_test
    y_test_global = y_test
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Feature importance
    print("\n📈 Feature Importances:")
    for feat, imp in sorted(zip(feature_columns, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"   {feat}: {imp:.4f}")
    
    return model, accuracy

# Train model on startup
print("\n" + "="*50)
print("🚀 TRAINING LEAD SCORING MODEL")
print("="*50)
try:
    model, accuracy = load_and_train_model()
    print("\n✅ API is ready to accept requests!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("⚠️ Please ensure 'lead_scoring_dataset.csv' exists in the same directory")

# --- API Request/Response Models ---
class LeadFeatures(BaseModel):
    deal_value: float
    calls: int
    emails: int
    meetings: int
    stage: str
    industry: str
    company_size: str

class PredictionResponse(BaseModel):
    probability: float
    score: int
    temperature: str
    reasons: List[str]
    nextAction: str
    confidence: float

# --- API Endpoints ---
@app.post("/predict-score", response_model=PredictionResponse)
async def predict_score(features: LeadFeatures):
    """Predict lead score based on deal features"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please check dataset.")
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            'deal_value': features.deal_value,
            'calls': features.calls,
            'emails': features.emails,
            'meetings': features.meetings,
            'stage': features.stage,
            'industry': features.industry,
            'company_size': features.company_size
        }])
        
        # Encode categorical variables
        for col in label_encoders:
            if col in input_data.columns:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
                except ValueError:
                    # Handle unknown category by using most frequent or 0
                    input_data[col] = 0
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training
        input_data = input_data[feature_columns]
        
        # Predict probability
        probability = model.predict_proba(input_data)[0][1]
        score = int(probability * 100)
        
        # Determine temperature
        if score >= 70:
            temperature = "Hot 🔥"
            action = "📞 Call immediately - High priority"
        elif score >= 40:
            temperature = "Warm 🌡️"
            action = "📧 Schedule demo meeting"
        else:
            temperature = "Cold ❄️"
            action = "✉️ Send nurturing email sequence"
        
        # Generate reasons based on feature importance
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-3:][::-1]
        
        reasons = []
        for idx in top_indices:
            feat_name = feature_columns[idx].replace('_', ' ').title()
            importance = importances[idx]
            if feat_name.lower() in ['stage', 'industry', 'company_size']:
                value = getattr(features, feat_name.lower().replace(' ', '_'))
                reasons.append(f"• {feat_name} = '{value}' (influence score: {importance:.1%})")
            else:
                value = getattr(features, feat_name.lower().replace(' ', '_'))
                reasons.append(f"• {feat_name} = {value} (influence score: {importance:.1%})")
        
        # Add overall reasoning
        reasons.append(f"• Historical data shows {probability:.1%} success rate for similar deals")
        
        # Calculate confidence based on prediction probability
        confidence = abs(probability - 0.5) * 2  # 0 to 1 scale
        
        return PredictionResponse(
            probability=round(probability, 4),
            score=score,
            temperature=temperature,
            reasons=reasons,
            nextAction=action,
            confidence=round(confidence, 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Lead Scoring ML API",
        "status": "active",
        "model_loaded": model is not None,
        "accuracy": accuracy if model else None,
        "endpoints": {
            "POST /predict-score": "Predict lead score",
            "GET /model-info": "Get model information",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_expected": len(feature_columns) if feature_columns else 0
    }

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate feature importance dictionary
    feature_importance = {
        feature_columns[i]: float(model.feature_importances_[i])
        for i in range(len(feature_columns))
    }
    
    # Sort by importance
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Get encoder information
    encoder_info = {}
    for col, encoder in label_encoders.items():
        encoder_info[col] = {
            "classes": encoder.classes_.tolist(),
            "num_classes": len(encoder.classes_)
        }
    
    # Calculate model performance if test data exists
    performance = {}
    if X_test_global is not None and y_test_global is not None:
        y_pred = model.predict(X_test_global)
        performance["accuracy"] = float(accuracy_score(y_test_global, y_pred))
    
    return {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "total_features": len(feature_columns),
        "feature_names": feature_columns,
        "feature_importance": feature_importance,
        "categorical_encoders": encoder_info,
        "performance": performance,
        "target_column": target_column
    }

@app.post("/retrain")
async def retrain_model():
    """Retrain the model with updated dataset"""
    try:
        global model, accuracy
        model, accuracy = load_and_train_model()
        return {
            "message": "Model retrained successfully",
            "accuracy": accuracy,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")