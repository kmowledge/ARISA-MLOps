def train_predict_plot_titanic():
    import os
    import zipfile
    from pathlib import Path
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dataset = "titanic"  # original competition dataset
    dataset_test = "wesleyhowe/titanic-labelled-test-set"  # test set augmented with target labels
    download_folder = Path("data/titanic")
    zip_path = download_folder / "titanic.zip"
    download_folder.mkdir(parents=True, exist_ok=True)

    api.competition_download_files(dataset, path=str(download_folder))
    api.dataset_download_files(dataset_test, path=str(download_folder), unzip=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(str(download_folder))

    os.remove(zip_path)

    import pandas as pd

    df_train = pd.read_csv(download_folder / "train.csv")
    df_ids = df_train.pop("PassengerId")  # set aside PassengerId
    df_train = df_train.drop(columns=["Ticket"])

    import re
    def extract_title(name):
        match = re.search(r',\s*([\w\s]+)\.', name)
        return match.group(1) if match else None
    df_train["Title"] = df_train["Name"].apply(extract_title)

    # pattern to match a letter followed by a number
    pattern = r'([A-Za-z]+)(\d+)'

    # run pattern on Cabin to extract all matches
    matches = df_train['Cabin'].str.extractall(pattern)
    matches.reset_index(inplace=True)

    # create a new column for each letter and number matched
    result = matches.pivot(index='level_0', columns='match', values=[0, 1])
    result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]

    # join to original train dataframe
    df_train = df_train.join(result[["0_0", "1_0"]])

    # fill nans
    df_train["1_0"] = df_train["1_0"].astype(float)
    df_train = df_train.fillna({"0_0": "N", "1_0": df_train["1_0"].mean()})
    df_train["1_0"] = df_train["1_0"].astype(int)

    # rename new columns and drop old ones
    df_train = df_train.rename(columns={"0_0": "Deck", "1_0": "CabinNumber"})
    df_train.drop(columns=["Cabin", "Name"], axis=1, inplace=True)

    df_train = df_train.fillna({"Embarked": "N", "Age": df_train["Age"].mean()})

    categorical = [
        "Pclass", 
        "Sex", 
        "Embarked",
        "Deck",
        "Title"
    ]

    y_train = df_train.pop("Survived")
    X_train = df_train

    categorical_indices = [X_train.columns.get_loc(col) for col in categorical if col in X_train.columns]

    import joblib
    import optuna
    from sklearn.model_selection import train_test_split
    from catboost import CatBoostClassifier, Pool, cv

    outfolder = Path("results")
    outfolder.mkdir(parents=True, exist_ok=True)

    best_params_path = outfolder / "best_params_v2.pkl"

    if not best_params_path.is_file():
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        
        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3),
                "iterations": trial.suggest_int("iterations", 50, 300),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 1),
                "random_strength": trial.suggest_float("random_strength", 1e-5, 100.0, log=True)
            }
            model = CatBoostClassifier(**params, verbose=0)
            model.fit(X_train_opt, y_train_opt, eval_set=(X_val_opt, y_val_opt), cat_features=categorical_indices, early_stopping_rounds=50)
            return model.get_best_score()["validation"]["Logloss"]
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        
        joblib.dump(study.best_params, best_params_path)
        params = study.best_params
    else:
        params = joblib.load(best_params_path)

    params["eval_metric"] = "F1"
    params["loss_function"] = "Logloss"

    model = CatBoostClassifier(
        **params,
        verbose=True
    )

    data = Pool(X_train, y_train, cat_features=categorical_indices)

    cv_results = cv(
        params=params,
        pool=data,
        fold_count=5,
        partition_random_seed=42,
        shuffle=True,
    )

    cv_results.to_csv(outfolder / "cv_results_v2.csv", index=False)

    import plotly.graph_objects as go

    # Create figure
    fig = go.Figure()

    # Add mean performance line
    fig.add_trace(
        go.Scatter(
            x=cv_results["iterations"], y=cv_results["test-F1-mean"], mode="lines", name="Mean F1 Score", line=dict(color="blue")
        )
    )

    # Add shaded error region
    fig.add_trace(
        go.Scatter(
            x=pd.concat([cv_results["iterations"], cv_results["iterations"][::-1]]),
            y=pd.concat([cv_results["test-F1-mean"]+cv_results["test-F1-std"], 
                        cv_results["test-F1-mean"]-cv_results["test-F1-std"]]),
            fill="toself", 
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            showlegend=False
        )
    )

    # Customize layout
    fig.update_layout(
        title="Cross-Validation (N=5) Mean F1 score with Error Bands",
        xaxis_title="Training Steps",
        yaxis_title="Performance Score",
        template="plotly_white",
        yaxis=dict(range=[0.5, 1])
    )

    fig.show()

    fig.write_image(outfolder / "test_f1_v2.png")

    import plotly.graph_objects as go

    # Create figure
    fig = go.Figure()

    # Add mean performance line
    fig.add_trace(
        go.Scatter(
            x=cv_results["iterations"], y=cv_results["test-Logloss-mean"], mode="lines", name="Mean logloss", line=dict(color="blue")
        )
    )

    # Add shaded error region
    fig.add_trace(
        go.Scatter(
            x=pd.concat([cv_results["iterations"], cv_results["iterations"][::-1]]),
            y=pd.concat([cv_results["test-Logloss-mean"]+cv_results["test-Logloss-std"], 
                        cv_results["test-Logloss-mean"]-cv_results["test-Logloss-std"]]),
            fill="toself", 
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            showlegend=False
        )
    )

    # Customize layout
    fig.update_layout(
        title="Cross-Validation (N=5) Mean Logloss with Error Bands",
        xaxis_title="Training Steps",
        yaxis_title="Logloss",
        template="plotly_white"
    )

    fig.write_image(outfolder / "test_logloss_v2.png")

    model.fit(
        X_train,
        y_train,
        verbose_eval=50,
        early_stopping_rounds=50,
        cat_features=categorical_indices,
        use_best_model=False,
        plot=True
    )

    model.save_model(outfolder / 'catboost_model_titanic_v2.cbm')
    joblib.dump(params, outfolder / 'model_params_v2.pkl')

    df_test = pd.read_csv(download_folder / "test.csv")
    df_test = df_test.drop(columns=["Ticket"])
    df_test_id = df_test.pop("PassengerId")
    df_test = df_test.fillna({"Embarked": "N", "Age": X_train["Age"].mean()})

    pattern = r'([A-Za-z]+)(\d+)'
    matches = df_test['Cabin'].str.extractall(pattern)
    matches.reset_index(inplace=True)
    result = matches.pivot(index='level_0', columns='match', values=[0, 1])
    result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]
    df_test = df_test.join(result[["0_0", "1_0"]])
    df_test["1_0"] = df_test["1_0"].astype(float)
    df_test = df_test.fillna({"0_0": "N", "1_0": X_train["CabinNumber"].mean()})
    df_test["1_0"] = df_test["1_0"].astype(int)
    df_test = df_test.rename(columns={"0_0": "Deck", "1_0": "CabinNumber"})

    df_test["Title"] = df_test["Name"].apply(extract_title)

    df_test.drop(columns=["Cabin", "Name"], axis=1, inplace=True)

    preds = model.predict(df_test[X_train.columns])

    import shap
    import matplotlib.pyplot as plt
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_test[X_train.columns])

    shap.summary_plot(shap_values, df_test, show=False)
    plt.savefig(outfolder / "test_shap_overall_v2.png")

    df_test["PassengerId"] = df_test_id
    df_test["Survived"] = preds
    df_test[["PassengerId", "Survived"]].to_csv(outfolder / "predictions_v2.csv", index=False)

if __name__ == "__main__":
    train_predict_plot_titanic()