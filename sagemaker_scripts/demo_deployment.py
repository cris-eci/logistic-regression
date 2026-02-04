#!/usr/bin/env python3
"""
Heart Disease Model - Deployment Demo
======================================

Run from SageMaker Code Editor terminal:
    python sagemaker_scripts/demo_deployment.py
"""
import json
import numpy as np
import logging

# Suppress SageMaker INFO messages (fixes hanging issue!)
logging.getLogger('sagemaker.config').setLevel(logging.WARNING)

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def main():
    print("=" * 70)
    print("🚀 HEART DISEASE MODEL - SAGEMAKER DEPLOYMENT")
    print("=" * 70)

    # Step 1: Initialize
    print("\n📦 Step 1: Initializing SageMaker session...")
    sagemaker_session = sagemaker.Session()
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    role = sagemaker.get_execution_role()
    print(f"   ✅ Region: {region}")
    print(f"   ✅ Bucket: {bucket}")
    print(f"   ✅ Role: {role[:50]}...")

    # Step 2: Upload to S3
    print("\n📤 Step 2: Uploading model.tar.gz to S3...")
    s3_model_path = sagemaker_session.upload_data(
        path='model.tar.gz',
        bucket=bucket,
        key_prefix='heart-disease-model'
    )
    print(f"   ✅ S3 Path: {s3_model_path}")

    # Step 3: Create Model object
    print("\n🔧 Step 3: Creating SageMaker Model object...")
    model = SKLearnModel(
        model_data=s3_model_path,
        role=role,
        entry_point='inference.py',
        source_dir='sagemaker_scripts',
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=sagemaker_session
    )
    print("   ✅ Model object created successfully")

    # Step 4: Deploy endpoint
    print("\n🌐 Step 4: Deploying endpoint...")
    print("   " + "-" * 50)
    print("   📋 Endpoint Name: heart-disease-prediction-endpoint")
    print("   📋 Instance Type: ml.t2.medium")
    print("   📋 Instance Count: 1")
    print(f"   📋 Model Data: {s3_model_path}")
    print("   📋 Framework: scikit-learn 1.2-1")
    print("   " + "-" * 50)
    
    try:
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            endpoint_name='heart-disease-prediction-endpoint'
        )
        print("   ✅ Endpoint deployed successfully!")
        
        # Step 5: Test the endpoint
        print("\n🧪 Step 5: Testing endpoint with sample patients...")
        
        test_cases = [
            {
                "name": "High-Risk Patient",
                "features": [65, 160, 320, 120, 2.5, 2],
                "description": "65yo, high BP, high cholesterol, low max HR"
            },
            {
                "name": "Low-Risk Patient", 
                "features": [35, 120, 180, 175, 0, 0],
                "description": "35yo, normal BP, low cholesterol, high max HR"
            },
            {
                "name": "Medium-Risk Patient",
                "features": [55, 140, 250, 150, 1.0, 1],
                "description": "55yo, borderline values"
            }
        ]
        
        for test in test_cases:
            response = predictor.predict({"features": test['features']})
            print(f"\n   📋 {test['name']}")
            print(f"      {test['description']}")
            print(f"      Features: {test['features']}")
            print(f"      Response: {response}")
        
        # Cleanup option
        print("\n" + "=" * 70)
        print("✅ DEPLOYMENT SUCCESSFUL!")
        print("=" * 70)
        print(f"\n📁 Endpoint ARN: {predictor.endpoint_name}")
        print("\n⚠️  Remember to delete the endpoint when done to avoid charges:")
        print("   predictor.delete_endpoint()")
        
    except Exception as e:
        print(f"\n   ❌ Deployment failed: {e}")
        
        # Fallback: Local inference test
        print("\n🧪 Step 5: Testing inference LOCALLY instead...")
        
        model_data = np.load('heart_disease_model.npy', allow_pickle=True).item()
        
        weights = model_data['weights']
        bias = model_data['bias']
        X_min = model_data['X_min']
        X_max = model_data['X_max']
        feature_names = model_data['feature_names']
        
        print(f"\n   Features: {feature_names}")
        
        test_cases = [
            {
                "name": "High-Risk Patient",
                "features": [65, 160, 320, 120, 2.5, 2],
                "description": "65yo, high BP, high cholesterol, low max HR"
            },
            {
                "name": "Low-Risk Patient", 
                "features": [35, 120, 180, 175, 0, 0],
                "description": "35yo, normal BP, low cholesterol, high max HR"
            },
            {
                "name": "Medium-Risk Patient",
                "features": [55, 140, 250, 150, 1.0, 1],
                "description": "55yo, borderline values"
            }
        ]
        
        for test in test_cases:
            features = np.array(test['features']).reshape(1, -1)
            features_norm = (features - X_min) / (X_max - X_min + 1e-8)
            z = np.dot(features_norm, weights) + bias
            probability = float(sigmoid(z)[0])
            
            if probability < 0.3:
                risk = "Low ✅"
            elif probability < 0.5:
                risk = "Moderate"
            elif probability < 0.7:
                risk = "High ⚠️"
            else:
                risk = "Very High 🚨"
            
            diagnosis = "Heart Disease ⚠️" if probability >= 0.5 else "No Heart Disease ✅"
            
            print(f"\n   📋 {test['name']}")
            print(f"      {test['description']}")
            print(f"      Features: {test['features']}")
            print(f"      Probability: {probability:.2%}")
            print(f"      Risk Level: {risk}")
            print(f"      Diagnosis: {diagnosis}")

        print("\n" + "=" * 70)
        print("📊 DEPLOYMENT DEMO COMPLETE (with local fallback)")
        print("=" * 70)
        print("\n📊 Summary:")
        print("   ┌─────────────────────────────────┬────────┐")
        print("   │ Component                       │ Status │")
        print("   ├─────────────────────────────────┼────────┤")
        print("   │ SageMaker Session               │   ✅   │")
        print("   │ Model uploaded to S3            │   ✅   │")
        print("   │ SageMaker Model object          │   ✅   │")
        print("   │ Inference script (inference.py) │   ✅   │")
        print("   │ Local inference test            │   ✅   │")
        print("   │ Real endpoint deployment        │   ❌   │")
        print("   └─────────────────────────────────┴────────┘")
        print(f"\n📁 Model artifacts stored in S3:")
        print(f"   {s3_model_path}")

if __name__ == "__main__":
    main()
