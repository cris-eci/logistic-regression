# AWS SageMaker Deployment Guide
## Heart Disease Prediction Model - AWS Academy Learner Lab

This guide walks you through deploying your trained logistic regression model to AWS SageMaker using the AWS Academy Learner Lab environment.

> ⚠️ **CRITICAL:** Your SageMaker Domain **MUST** be configured with **"Public internet only"** network access. If your domain is in VPC-only mode, you will experience connection timeouts and the deployment will fail. See [Changing VPC to Public](#changing-vpc-to-public) if you need to fix this.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Learner Lab Limitations](#learner-lab-limitations)
3. [Step 1: Prepare Your Model Locally](#step-1-prepare-your-model-locally)
4. [Step 2: Start AWS Learner Lab](#step-2-start-aws-learner-lab)
5. [Step 3: Create SageMaker Domain](#step-3-create-sagemaker-domain)
6. [Step 4: Create User Profile](#step-4-create-user-profile)
7. [Step 5: Create Code Editor Space](#step-5-create-code-editor-space)
8. [Step 6: Upload Project Files](#step-6-upload-project-files)
9. [Step 7: Run Deployment Demo](#step-7-run-deployment-demo)
10. [Step 8: Capture Screenshots](#step-8-capture-screenshots)
11. [Changing VPC to Public](#changing-vpc-to-public)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- ✅ Completed Jupyter notebook (`heart_disease_lr_analysis.ipynb`)
- ✅ Trained model file (`heart_disease_model.npy`)
- ✅ Model package (`model.tar.gz`)
- ✅ Deployment scripts in `sagemaker_scripts/` folder
- ✅ AWS Academy Learner Lab access

### Required Files Structure

```
logistic-regression/
├── heart_disease_lr_analysis.ipynb   # Your completed notebook
├── heart_disease_model.npy           # Trained model
├── model.tar.gz                      # Packaged model for SageMaker
├── dataset/
│   └── Heart_Disease_Prediction.csv
└── sagemaker_scripts/
    ├── inference.py                  # SageMaker inference handler
    └── demo_deployment.py            # Deployment demo script
```

---

## Learner Lab Limitations

⚠️ **Important:** AWS Academy Learner Labs have IAM restrictions that block certain actions:

| Action | Status | Notes |
|--------|--------|-------|
| Create SageMaker Domain | ✅ Allowed | |
| Create Code Editor Space | ✅ Allowed | |
| Upload model to S3 | ✅ Allowed | |
| Create SageMaker Model | ✅ Allowed | |
| **Create Endpoint** | ❌ **Blocked** | IAM policy restriction |
| **Create EndpointConfig** | ❌ **Blocked** | IAM policy restriction |

**Solution:** Our `demo_deployment.py` script demonstrates the full workflow and tests inference locally, working around the endpoint restriction.

---

## Step 1: Prepare Your Model Locally

### 1.1 Verify Model Files Exist

In your local terminal, check that all required files are present:

```bash
cd /path/to/logistic-regression
ls -la
```

You should see:
```
heart_disease_model.npy
model.tar.gz
sagemaker_scripts/
```

### 1.2 Verify model.tar.gz Contents

```bash
tar -tzvf model.tar.gz
```

Expected output:
```
heart_disease_model.npy
sagemaker_scripts/inference.py
```

### 1.3 Test Locally First (Optional)

```bash
python sagemaker_scripts/demo_deployment.py
```

This will run in local mode and verify your model works before uploading to AWS.

---

## Step 2: Start AWS Learner Lab

1. Go to your **AWS Academy** course in Canvas/LMS
2. Click on **Modules** → **Learner Lab**
3. Click **Start Lab** (green button)
4. Wait for the lab status to show **🟢 Ready** (takes 1-2 minutes)
5. Click **AWS** (with green dot) to open the AWS Console

> ⏱️ **Note:** Learner Lab sessions last 4 hours. Save your work!

---

## Step 3: Create SageMaker Domain

### 3.1 Navigate to SageMaker

1. In the AWS Console search bar, type **SageMaker**
2. Click **Amazon SageMaker**

### 3.2 Create Domain

1. In the left sidebar, click **Admin configurations** → **Domains**
2. Click **Create domain**
3. Select **Set up for organizations** → Click **Set up**

### 3.3 Configure Domain Details

**Page 1 - Domain details:**
- **Domain name:** `heart-disease-domain` (or any name you prefer)
- **Authentication:** Keep default (Login through IAM)
- Click **Next**

**Page 2 - Roles:**
- Select **Use an existing role**
- **Default execution role:** Select `LabRole` from the dropdown
- Click **Next**

**Page 3 - Applications:**
- **SageMaker Studio:** Select **SageMaker Studio - New**
- **Code Editor:** 
  - ✅ Enable it
  - Set idle shutdown to **60 minutes**
- Click **Next**

**Page 4 - Network:**

> 🚨 **THIS IS THE MOST CRITICAL STEP!**

- ✅ Select **"Public internet only"** ← **REQUIRED!**
- ❌ Do NOT select "VPC only" - this will cause connection timeouts!
- **VPC:** Select the default VPC
- **Subnets:** Select at least **2 subnets** from different availability zones
- **Security group:** Select the **default** security group
- Click **Next**

> ⚠️ **If you already created a domain with VPC-only**, you need to delete it and recreate with "Public internet only". See [Changing VPC to Public](#changing-vpc-to-public).

**Page 5 - Review:**
- Review your settings
- Click **Submit**

### 3.4 Wait for Domain Creation

⏱️ **This takes 5-8 minutes.** The status will show "Creating" → "InService"

You can monitor progress in the **Domains** list.

---

## Step 4: Create User Profile

Once the domain is ready (status: **InService**):

1. Click on your domain name (e.g., `heart-disease-domain`)
2. Go to the **User profiles** tab
3. Click **Add user**

### Configure User:

- **User profile name:** `student` (or your name)
- **Execution role:** Select `LabRole`
- Click **Next** through all remaining pages
- Click **Submit**

⏱️ Wait 1-2 minutes for the user profile to be created.

---

## Step 5: Create Code Editor Space

### 5.1 Open SageMaker Studio

1. In your domain, find your user profile
2. Click **Launch** → **Studio**

### 5.2 Create Code Editor Space

1. In SageMaker Studio, look at the left sidebar
2. Click **Applications** (or find "Code Editor" in the home page)
3. Click **Create Code Editor space**

### Configure Space:

- **Name:** `ml-workspace`
- **Instance type:** `ml.t3.medium` (sufficient and cost-effective)
- **Storage:** Keep default (5 GB)
- Click **Create space**

### 5.3 Run the Space

1. After creation, click **Run space**
2. Wait 2-3 minutes for the space to start
3. When ready, click **Open Code Editor**

> 🎉 You now have VS Code running in AWS!

---

## Step 6: Upload Project Files

### 6.1 Open File Explorer

In Code Editor (VS Code in browser):
1. Click the **Explorer** icon (📁) in the left sidebar
2. You'll see an empty workspace

### 6.2 Upload Files

**Option A: Drag and Drop**
1. Open your local file manager
2. Select these files/folders:
   - `model.tar.gz`
   - `heart_disease_model.npy`
   - `sagemaker_scripts/` folder
3. Drag them into the Code Editor file explorer

**Option B: Using Terminal**
1. Open Terminal: **Terminal** → **New Terminal**
2. Use the upload feature in VS Code

### 6.3 Verify Upload

In the Code Editor terminal:

```bash
ls -la
```

You should see:
```
model.tar.gz
heart_disease_model.npy
sagemaker_scripts/
```

---

## Step 7: Run Deployment Demo

### 7.1 Open Terminal

In Code Editor: **Terminal** → **New Terminal**

### 7.2 Run the Demo Script

```bash
python sagemaker_scripts/demo_deployment.py
```

### 7.3 Expected Output

```
======================================================================
🚀 HEART DISEASE MODEL - DEPLOYMENT DEMO
======================================================================

📋 Step 1: Checking required files...
   ✅ Found: model.tar.gz

📦 Step 2: Initializing SageMaker session...
   ✅ Region: us-east-1
   ✅ Bucket: sagemaker-us-east-1-XXXXXXXXXXXX
   ✅ Role: arn:aws:iam::XXXXXXXXXXXX:role/LabRole...

📤 Step 3: Uploading model.tar.gz to S3...
   ✅ S3 Path: s3://sagemaker-us-east-1-XXXX/heart-disease-model/model.tar.gz

🔧 Step 4: Creating SageMaker Model object...
   ✅ Model object created successfully

   ⚠️  LEARNER LAB LIMITATION:
   Endpoint creation is blocked by IAM policy.

🧪 Step 5: Testing inference LOCALLY (simulating endpoint)...

   Feature order: ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Number of vessels fluro']
------------------------------------------------------------

   📋 High-Risk Patient
      65yo, high BP, high cholesterol, low max HR, high ST depression, 3 vessels
      Input: [65, 160, 350, 120, 3.5, 3]
      Probability: 99.38%
      Diagnosis: Heart Disease (HIGH ⚠️)

   📋 Low-Risk Patient
      35yo, normal BP, low cholesterol, high max HR, no ST depression, 0 vessels
      Input: [35, 120, 180, 170, 0.0, 0]
      Probability: 0.28%
      Diagnosis: No Heart Disease (LOW ✅)

   📋 Medium-Risk Patient
      55yo, borderline BP, moderate cholesterol, normal HR, some ST depression
      Input: [55, 140, 250, 145, 1.5, 1]
      Probability: 45.23%
      Diagnosis: No Heart Disease (MEDIUM)

======================================================================
✅ DEPLOYMENT DEMO COMPLETE!
======================================================================
```

---

## Step 8: Capture Screenshots

For your homework submission, capture these screenshots:

### Required Screenshots

| Screenshot | What to Capture |
|------------|-----------------|
| 1. SageMaker Domain | Domain list showing your domain "InService" |
| 2. Code Editor Space | Your running Code Editor space |
| 3. Uploaded Files | File explorer showing `model.tar.gz` and scripts |
| 4. Deployment Output | Terminal showing successful deployment demo |
| 5. S3 Upload | S3 console showing uploaded model (optional) |

### How to Take Screenshots

**In Code Editor:**
- Use your OS screenshot tool (PrtScn on Windows, Cmd+Shift+4 on Mac)
- Or use browser extensions

**Save Screenshots:**
- Save to your project folder as:
  - `screenshots/01_sagemaker_domain.png`
  - `screenshots/02_code_editor_space.png`
  - `screenshots/03_uploaded_files.png`
  - `screenshots/04_deployment_output.png`

---

## Changing VPC to Public

🚨 **If your domain was created with "VPC only" network access, you MUST change it to "Public internet only".**

Symptoms of VPC-only domain:
- Script hangs at "Initializing SageMaker session..."
- Connection timeout errors: `ConnectTimeoutError... sts.us-east-1.amazonaws.com`
- Unable to reach AWS services

### Option 1: Delete and Recreate Domain (Recommended)

1. Go to **SageMaker** → **Admin configurations** → **Domains**
2. Select your domain
3. Click **Edit** to check network settings
4. If it shows "VPC only", you need to delete and recreate:

**Steps to delete:**
1. First, delete all **User Profiles** in the domain
2. Delete all **Spaces** (Code Editor, Studio apps)
3. Then delete the **Domain** itself
4. Wait 5-10 minutes for complete deletion
5. **Recreate** the domain with **"Public internet only"**

### Option 2: Edit Domain Network (May Not Work)

Some domains allow network editing:
1. Go to domain → **Domain settings** → **Network**
2. Click **Edit**
3. Change to **"Public internet only"**
4. Save changes

> ⚠️ This option may not be available for all domains. If grayed out, use Option 1.

### Verifying Your Domain is Public

1. Go to your domain in SageMaker
2. Click **Domain settings**
3. Look for **Network** section
4. It should say **"Public internet only"**

If it says "VPC only", your scripts will NOT work until you fix this.

---

## Troubleshooting

### Error: "ConnectTimeoutError" or Script Hangs

```
ConnectTimeoutError... Connect timeout on endpoint URL: sts.us-east-1.amazonaws.com
```

**Cause:** Your domain is configured with "VPC only" network, which has no internet access.  
**Solution:** Delete your domain and recreate with **"Public internet only"**. See [Changing VPC to Public](#changing-vpc-to-public).

### Error: "AccessDeniedException on CreateEndpointConfig"

```
User is not authorized to perform: sagemaker:CreateEndpointConfig
```

**Cause:** Learner Lab blocks endpoint creation.  
**Solution:** This is expected! The demo script tests locally instead.

### Error: "No module named 'sagemaker'"

```
ModuleNotFoundError: No module named 'sagemaker'
```

**Solution:** Install the SDK:
```bash
pip install sagemaker boto3
```

### Error: "model.tar.gz not found"

**Solution:** Make sure you uploaded the file. If you only have `heart_disease_model.npy`, the demo script will create `model.tar.gz` automatically.

### Error: Script Hangs at "Initializing SageMaker session"

**Cause:** Domain is in VPC-only mode without internet access.  
**Solution:** Delete your domain and recreate with **"Public internet only"**. See [Changing VPC to Public](#changing-vpc-to-public).

### Domain Creation Stuck

If domain creation takes more than 15 minutes:
1. Delete the domain
2. Try again with **"Public internet only"** option ← **Critical!**
3. Ensure you selected at least 2 subnets
4. Make sure you're using the **default VPC** and **default security group**

### Session Expired

If your Learner Lab session expires:
1. Return to your AWS Academy course
2. Click **Start Lab** again
3. Your SageMaker domain persists between sessions

---

## Clean Up (Important!)

⚠️ **Before ending your lab session:**

1. **Stop Code Editor Space:**
   - Go to SageMaker Studio → Applications
   - Find your Code Editor space
   - Click **Stop**

2. **Stop Running Apps:**
   - Check for any running applications
   - Stop all to avoid charges

> 💡 Learner Lab sessions don't charge real money, but it's good practice!

---

## Summary

| Step | Action | Time |
|------|--------|------|
| 1 | Prepare model locally | 5 min |
| 2 | Start Learner Lab | 2 min |
| 3 | Create SageMaker Domain | 8 min |
| 4 | Create User Profile | 2 min |
| 5 | Create Code Editor Space | 3 min |
| 6 | Upload project files | 2 min |
| 7 | Run deployment demo | 1 min |
| 8 | Capture screenshots | 5 min |
| **Total** | | **~30 min** |

---

## What You Demonstrated

Even with Learner Lab limitations, you successfully:

1. ✅ Created a SageMaker Domain and User Profile
2. ✅ Launched a Code Editor (VS Code) in the cloud
3. ✅ Uploaded your trained model to S3
4. ✅ Created a SageMaker Model object
5. ✅ Tested inference predictions

This demonstrates understanding of the full ML deployment pipeline, even if the actual endpoint couldn't be created due to IAM restrictions.

---

## Appendix: Full Deployment (Non-Learner Lab)

If you have a full AWS account (not Learner Lab), you can:

```bash
# Create real endpoint (takes 3-5 minutes)
python sagemaker_scripts/deploy.py

# Test the endpoint
python sagemaker_scripts/test_endpoint.py

# CRITICAL: Delete endpoint to stop charges ($0.05/hour)
python sagemaker_scripts/cleanup.py
```

The `deploy.py`, `test_endpoint.py`, and `cleanup.py` scripts would need to be created for this purpose.
