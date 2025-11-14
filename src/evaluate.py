import argparse, joblib, pandas as pd
from sklearn.metrics import accuracy_score

def evaluate(data_path, model_path, target):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]
    model = joblib.load(model_path)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"✅ Evaluation complete! Accuracy: {acc:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    evaluate(args.data_path, args.model_path, args.target)
