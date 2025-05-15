# decision_tree_housing.py
"""
California Housing Price Prediction with Decision Tree
- Loads California housing dataset
- Trains Decision Tree Regressor
- Visualizes the tree
- Saves results and metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor, plot_tree


RANDOM_STATE = 42
OUTPUT_DIR = Path("output")
FEATURES = ["HouseAge", "Population"]  # Features to analyze


def create_output_dir():
    """Create output directory if not exists"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "data").mkdir(exist_ok=True)


def load_data():
    """Load and prepare California housing data"""
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df[FEATURES + ["target"]].head(5)  # Use first 5 samples to see how decision tree is learning 


def train_model(X, y):
    """Train and return Decision Tree model"""
    model = DecisionTreeRegressor(
        random_state=RANDOM_STATE, max_depth=3  
    )
    model.fit(X, y)
    return model


def visualize_tree(model, feature_names, filename):
    """Save tree visualization to file"""
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True)
    plt.savefig(OUTPUT_DIR / "images" / filename, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_metrics(y_true, y_pred):
    """Calculate and return MSE metrics"""
    return np.mean(np.square(y_true - y_pred))


def save_metrics(metrics, filename):
    """Save metrics to JSON file"""
    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    
    create_output_dir()

    # Load data
    df = load_data()
    df.to_csv(OUTPUT_DIR / "data" / "processed_data.csv", index=False)

    X = df[FEATURES]
    y = df["target"]

    # Train model
    model = train_model(X, y)

    # Visualize
    visualize_tree(model, FEATURES, "decision_tree.png")

    # Calculate metrics
    metrics = {
        "root": {"mean": float(y.mean()), "mse": float(calculate_metrics(y, y.mean()))},
        "left_split": {
            "mean": float(df[df["Population"] <= 409]["target"].mean()),
            "mse": float(
                calculate_metrics(
                    df[df["Population"] <= 409]["target"],
                    df[df["Population"] <= 409]["target"].mean(),
                )
            ),
        },
        "right_split": {
            "mean": float(df[df["Population"] > 409]["target"].mean()),
            "mse": float(
                calculate_metrics(
                    df[df["Population"] > 409]["target"],
                    df[df["Population"] > 409]["target"].mean(),
                )
            ),
        },
        "feature_importances": dict(zip(FEATURES, model.feature_importances_.tolist())),
    }

    # Save results
    save_metrics(metrics, "metrics.json")

    print("Analysis completed successfully!")
    print(f"Results saved to {OUTPUT_DIR} directory")


if __name__ == "__main__":
    main()
