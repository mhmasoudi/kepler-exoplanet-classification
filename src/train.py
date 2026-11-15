
from pathlib import Path
from scipy.stats import randint

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "kepler_binary.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "kepler_xgboost.joblib"

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target", "kepid"])
    y = df["target"]
    groups = df["kepid"]

    return X, y, groups


def split_data(X, y, groups):
    split_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.40,
        random_state=RANDOM_STATE
    )

    train_idx, temp_idx = next(
        split_1.split(X, y, groups=groups)
    )

    split_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    val_relative_idx, test_relative_idx = next(
        split_2.split(
            X.iloc[temp_idx],
            y.iloc[temp_idx],
            groups=groups.iloc[temp_idx]
        )
    )

    val_idx = temp_idx[val_relative_idx]
    test_idx = temp_idx[test_relative_idx]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(X_train):
    numeric_features = X_train.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric_features
            )
        ]
    )

    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=1
        ))
    ])

    return xgb_pipeline


def tune_model(xgb_pipeline, X_train, y_train, groups):
    param_distributions = {
        "classifier__n_estimators": randint(100, 501),
        "classifier__max_depth": randint(3, 10),
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__min_child_weight": randint(1, 8)
    }

    cv = GroupKFold(n_splits=5)

    random_search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train, groups=groups)

    print("Best parameters:", random_search.best_params_)
    print("Best cross-validation F1:", random_search.best_score_)

    return random_search.best_estimator_


def save_model(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


def main():
    print("Loading dataset...")
    X, y, groups = load_data()

    print("Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, groups
    )

    print(f"Training observations: {len(X_train)}")
    print(f"Validation observations: {len(X_val)}")
    print(f"Test observations: {len(X_test)}")

    print("\nBuilding model...")
    xgb_pipeline = build_model(X_train)

    print("\nTuning hyperparameters...")
    best_model = tune_model(
        xgb_pipeline,
        X_train,
        y_train,
        groups.iloc[X_train.index]
    )

    save_model(best_model)


if __name__ == "__main__":
    main()
