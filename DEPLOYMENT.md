# 🚀 Deployment Guide

## Option 1: Railway (Recommended)

### Step 1: Prepare Repository
```bash
# Make sure all files are committed
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway will automatically detect it's a Python app
7. Wait for deployment (2-3 minutes)

### Step 3: Configure Environment
- Railway will automatically set `PORT` environment variable
- No additional configuration needed

### Step 4: Access Your App
- Railway will provide a URL like: `https://your-app-name.railway.app`
- Your dashboard will be available at this URL

---

## Option 2: Render

### Step 1: Prepare Repository
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Deploy to Render
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: `ecommerce-dashboard`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api.fastapi_service:app --host 0.0.0.0 --port $PORT`
6. Click "Create Web Service"

### Step 3: Access Your App
- Render will provide a URL like: `https://ecommerce-dashboard.onrender.com`

---

## Option 3: Heroku

### Step 1: Install Heroku CLI
```bash
# Install Heroku CLI (if not already installed)
# Visit: https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Deploy to Heroku
```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-app-name

# Deploy
git push heroku main

# Open app
heroku open
```

---

## Environment Variables

No environment variables are required for basic deployment. The app uses fixed data files.

## Data Files

The following data files are included in deployment:
- `datasets/processed/sales_data_jan_june_2025.csv`
- `datasets/processed/returns_jan_june_2025.csv`
- `datasets/processed/historical_data_2018_nov2024.csv`
- `cache/historical_patterns_cache.json`

## Troubleshooting

### Common Issues:
1. **Port binding error**: Make sure to use `$PORT` environment variable
2. **Module not found**: Ensure `PYTHONPATH` is set correctly
3. **Data file not found**: Check file paths in `fastapi_service.py`

### Debug Commands:
```bash
# Check if app starts locally
python3 start_dashboard.py

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/analysis
```

## Performance Notes

- **Cold starts**: May take 10-30 seconds on first request
- **Memory usage**: ~200-500MB depending on data size
- **Response time**: <2 seconds for most API calls
- **Concurrent users**: Supports 10-50 concurrent users on free tiers

## Cost

- **Railway**: Free tier includes 500 hours/month
- **Render**: Free tier includes 750 hours/month
- **Heroku**: Requires credit card, $7/month for basic plan

---

**Recommended: Railway for easiest deployment with best free tier!** 🚂
