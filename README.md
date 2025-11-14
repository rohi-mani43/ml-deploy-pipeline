# ML Deployment Pipeline (Iris Dataset)

This project trains, evaluates, and deploys a simple RandomForestClassifier model 
on the Iris dataset using Python, Flask, Docker, and GitHub Actions.

### 🧠 Dataset
- **File**: data/IRIS.csv
- **Target column**: `species`

### ⚙️ Steps to run (Windows PowerShell)

1. Open PowerShell in this folder:
```powershell
cd ml_deploy_pipeline
```

2. Create virtual environment:
```powershell
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```powershell
pip install -r src\requirements.txt
pip install -r app\requirements.txt
```

4. Train model:
```powershell
python src\train.py --data-path data\IRIS.csv --model-path artifacts\model.joblib --target species
```

5. Evaluate:
```powershell
python src\evaluate.py --data-path data\IRIS.csv --model-path artifacts\model.joblib --target species
```

6. Run Flask API:
```powershell
cd app
python app.py
```

Then open your browser at **http://localhost:5000**  
For predictions:
```powershell
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{'instances': [[5.1,3.5,1.4,0.2]]}"
```
