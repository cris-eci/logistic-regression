"""
SageMaker Inference Handler for Heart Disease Prediction Model
This file handles model loading, input parsing, prediction, and output formatting.
"""

import numpy as np
import json
import os


def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def model_fn(model_dir):
    """
    Load the trained model from disk.
    Called once when the endpoint starts.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Dictionary containing model weights, bias, and preprocessing params
    """
    model_path = os.path.join(model_dir, "heart_disease_model.npy")
    model = np.load(model_path, allow_pickle=True).item()
    print(f"Model loaded successfully. Features: {model['feature_names']}")
    return model


def input_fn(request_body, content_type='application/json'):
    """
    Parse incoming request data.
    
    Args:
        request_body: Raw request body
        content_type: Content type of the request
        
    Returns:
        NumPy array of features
    """
    if content_type == 'application/json':
        data = json.loads(request_body)
        features = np.array(data['features']).reshape(1, -1)
        return features
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Make predictions using the loaded model.
    
    Args:
        input_data: NumPy array of features (1, n_features)
        model: Loaded model dictionary
        
    Returns:
        Dictionary with probability, prediction, and risk level
    """
    # Extract model components
    w = model['weights']
    b = model['bias']
    X_min = model['X_min']
    X_max = model['X_max']
    
    # Normalize input using training set statistics
    X_norm = (input_data - X_min) / (X_max - X_min + 1e-8)
    
    # Compute prediction
    z = np.dot(X_norm, w) + b
    probability = sigmoid(z)
    prediction = (probability >= 0.5).astype(int)
    
    # Determine risk level
    prob_value = float(probability[0])
    if prob_value >= 0.7:
        risk_level = 'HIGH'
    elif prob_value >= 0.4:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    return {
        'probability': prob_value,
        'prediction': int(prediction[0]),
        'risk_level': risk_level
    }


def output_fn(prediction, accept='application/json'):
    """
    Format the prediction response.
    
    Args:
        prediction: Prediction dictionary
        accept: Accepted content type
        
    Returns:
        JSON string of the prediction
    """
    if accept == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
