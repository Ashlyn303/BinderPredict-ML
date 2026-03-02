# Deployment Guide - Streamlit Community Cloud

Follow these steps to host your **BinderPredict-ML Dashboard** permanently on the web. Once deployed, others can view and use the app, but they **cannot edit your code**.

## Step 1: Prepare GitHub
You have already pushed your code to: `https://github.com/Ashlyn303/BinderPredict-ML`.
Make sure the latest model files (in `pytorch_results/`, etc.) are tracked and pushed.

## Step 2: Create a Streamlit Cloud Account
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **"Connect GitHub account"** and sign in with your GitHub credentials.

## Step 3: Deploy the App
1. On your Streamlit Dashboard, click **"New app"**.
2. **Repository**: Select `Ashlyn303/BinderPredict-ML`.
3. **Branch**: `main`.
4. **Main file path**: `dashboard_app.py`.
5. Click **"Deploy!"**.

## Step 4: Access & Share
*   Your app will be live at a URL like `https://binderpredict-ml.streamlit.app`.
*   You can share this link with anyone. 
*   **Security**: Streamlit Cloud provides a "View Only" environment. Users can interact with the widgets, but they cannot access or change your underlying repository files.

---

### Troubleshooting
*   **Missing Dependencies**: If the app fails to start, ensure `requirements.txt` contains all necessary libraries (it currently does).
*   **Large Files**: If your model files are extremely large (over 100MB), you might need to use **Git LFS**, but for this project (under 10MB), standard Git is fine.
