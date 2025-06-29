# Python Sentiment Analysis Server Troubleshooting

## Common Issues and Solutions

### Issue 1: "Not Found" at localhost:3000
**Problem**: The Python server is not running on the correct port or not starting properly.

**Solutions**:

#### Check 1: Verify Python Server is Running
```bash
# In python_backend directory
python sentiment_analyzer.py
```

You should see output like:
```
Starting Advanced Sentiment Analysis API...
Available methods: VADER=True, TextBlob=True, spaCy=False, FinBERT=False, BERT=False
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://[::1]:5000
```

#### Check 2: Correct Port
The Python server runs on **port 5000**, not 3000.
- Python API: `http://localhost:5000`
- React frontend: `http://localhost:3000`

#### Check 3: Test API Directly
Open your browser and go to:
```
http://localhost:5000/health
```

You should see:
```json
{
  "status": "healthy",
  "available_methods": {
    "vader": true,
    "textblob": true,
    "spacy": false,
    "finbert": false,
    "bert": false
  }
}
```

### Issue 2: Dependencies Not Installed
**Problem**: Missing Python packages

**Solution**:
```bash
cd python_backend
pip install -r requirements.txt
```

### Issue 3: Python Version Compatibility
**Problem**: Using incompatible Python version

**Solution**:
```bash
# Check Python version (should be 3.8+)
python --version

# If using Python 3.13, some packages might need updates
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

### Issue 4: Port Already in Use
**Problem**: Port 5000 is occupied

**Solution**:
```bash
# Kill process on port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F

# Kill process on port 5000 (Mac/Linux)
lsof -ti:5000 | xargs kill -9

# Or use a different port
python sentiment_analyzer.py --port 5001
```

### Issue 5: CORS Issues
**Problem**: Frontend can't connect to backend

**Solution**: The server includes CORS headers, but if issues persist:
```python
# In sentiment_analyzer.py, update CORS configuration
CORS(app, origins=["http://localhost:3000"])
```

## Step-by-Step Debugging

### Step 1: Check Directory Structure
```
python_backend/
├── sentiment_analyzer.py
├── requirements.txt
├── setup.py
└── README.md
```

### Step 2: Install Dependencies
```bash
cd python_backend
pip install flask flask-cors numpy pandas vaderSentiment textblob
```

### Step 3: Start Server with Verbose Output
```bash
python sentiment_analyzer.py
```

### Step 4: Test API Endpoints
```bash
# Test health endpoint
curl http://localhost:5000/health

# Test analysis endpoint
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a positive news about stocks rising"}'
```

### Step 5: Check Frontend Connection
In your React app, the NewsAnalyzer component should show:
- ✅ Green "API Connected" status
- Available methods listed
- No error messages

## Quick Fix Commands

### Minimal Setup (if full requirements fail)
```bash
cd python_backend
pip install flask flask-cors vaderSentiment textblob
python sentiment_analyzer.py
```

### Alternative Port
If port 5000 is busy, modify the last line in `sentiment_analyzer.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)
```

Then update frontend to use port 5001.

### Windows-Specific Issues
```bash
# Use python3 if python doesn't work
python3 sentiment_analyzer.py

# Install in user directory if permission issues
pip install --user flask flask-cors vaderSentiment textblob
```

## Expected Behavior

### When Working Correctly:
1. **Python Terminal**: Shows "Running on http://127.0.0.1:5000"
2. **Browser at localhost:5000/health**: Shows JSON response
3. **React App**: Shows green "API Connected" status
4. **Analysis**: Returns detailed sentiment results

### Common Error Messages:
- `ModuleNotFoundError`: Install missing packages
- `Address already in use`: Change port or kill existing process
- `Permission denied`: Use `--user` flag with pip
- `Connection refused`: Server not running

## Testing the Complete Flow

1. **Start Python server**:
   ```bash
   cd python_backend
   python sentiment_analyzer.py
   ```

2. **Verify server is running**:
   ```bash
   curl http://localhost:5000/health
   ```

3. **Start React frontend** (in another terminal):
   ```bash
   npm run dev
   ```

4. **Test in browser**:
   - Go to `http://localhost:3000`
   - Click "News Analyzer" tab
   - Should show "API Connected" status
   - Paste some text and click "Analyze with AI"

If you're still having issues, please share:
1. The exact error message from the Python terminal
2. Your Python version (`python --version`)
3. Operating system
4. Any error messages in the browser console