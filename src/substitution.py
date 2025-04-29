import sys
import tensorflow as tf
from AMN import r2_score, evaluate_model
from ANN_comparison import build_ann, fit_model, get_data
from sklearn.model_selection import KFold


def ann_comparaison(
    filename, regression, cross_val=5, n_hidden_dim=1000, mid_layer=49, untrain_seed=0
):
    metric_1_list = []  # Q2 for regression Acc for classification
    metric_2_list = []  # Q2 variance weighted for regression AUC for classification
    folder = "Regressions" if regression else "Classifications"

    X, Y = get_data(folder + "./data/raw/" + filename + ".csv")
    tf.keras.utils.set_random_seed(10)  # Same seed for kfold
    kf = KFold(n_splits=cross_val, shuffle=True, random_state=42)
    folds = kf.split(X)
    for train_index, val_index in folds:
        X_fold_train, X_fold_val = X[train_index], X[val_index]
        y_fold_train, y_fold_val = Y[train_index], Y[val_index]

        input_dimension = X_fold_train.shape[1]
        output_dimension = len(y_fold_train[0])

        # Build the model
        tf.keras.utils.set_random_seed(
            untrain_seed
        )  # Different seed for network initialization
        ann_model = build_ann(
            input_dim=input_dimension,
            regression=regression,
            mid_layer=mid_layer,
            output_dim=output_dimension,
            neurons_per_layer=n_hidden_dim,
        )

        # Fit model
        if regression:
            fit_model(
                ann_model,
                X_fold_train,
                y_fold_train,
                optimizer="adam",
                loss="mse",
                metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
            )
        else:
            fit_model(
                ann_model,
                X_fold_train,
                y_fold_train,
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=[
                    "accuracy",
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC(),
                ],
            )

        # Evaluate model
        if regression:
            y_pred, loss, q2 = evaluate_model(ann_model, X_fold_val, y_fold_val)
            metric_1_list.append(r2_score(y_fold_val, y_pred))
            metric_2_list.append(
                r2_score(y_fold_val, y_pred, multioutput="variance_weighted")
            )
        else:
            evaluation = ann_model.evaluate(X_fold_val, y_fold_val, verbose=0)
            metric_1_list.append(evaluation[1])  # Accuracy
            metric_2_list.append(evaluation[4])  # AUC

    return metric_1_list, metric_2_list

if __name__ == "__main__":
    if sys.argv[1] not in ["regression", "classification"]:
        raise ValueError("First argument must be 'regression' or 'classification'")
    task = sys.argv[1]  # 'regression' or 'classification'

    if task == "regression":
        regression = True
        dataset_list = ["naval_propulsion_plant",	"white_wine",	"cpu_activity",	"kin8nm",	"miami_housing",	"energy_efficiency",	"QSAR_fish_toxicity"]
    else:
        regression = False
        dataset_list = ["diabetes", "spambase", "pc4", "pc3", "kc2", "kc1", "pc1", "Bioresponse", "wdbc", "phoneme" "qsar-biodeg", "madelon", "ozone-level-8hr", "banknote-authentication", "blood-transfusion-service-center", "climate-model-simulation-crashes", "wilt", "numerai28.6"]
    
    dict_score_amn = {}
    dict_score_ANN = {}

    cross_val = 5
    list_hidden_dim = [1, 10, 25, 50, 100, 250, 500, 700]
    list_replicats = []
    for rand_seed in range(20): # 20 replicates
        dict_score_ANN = {}
        for filename in dataset_list:
            q2_list_ANN = []
            for n_hidden_dim in list_hidden_dim:
                cross_vall_ann, _ = ann_comparaison(filename,
                                                    regression=regression,
                                                    cross_val=cross_val,
                                                    n_hidden_dim=n_hidden_dim)
                q2_list_ANN.append([cross_vall_ann])

            dict_score_ANN[filename] = q2_list_ANN
        list_replicats.append(dict_score_ANN)