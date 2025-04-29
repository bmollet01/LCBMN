import tensorflow as tf
import numpy as np
import pandas
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer


def read_XY(filename, nY=1):
    # Format data for training
    # Function read_training_data is defined in module (1)
    # if scaling == 'X' X is scaled
    # if scaling == 'Y' Y is scaled
    # if scaling == 'XY' X and Y are scaled
    _, XY = read_csv(filename)
    XY = np.asarray(XY)
    X = XY[:, :-nY]
    Y = XY[:, -nY:]
    return X, Y


def read_csv(filename):
    # Reading datafile with pandas
    # Return HEADER and DATA
    if filename[-3:] != "csv":
        filename += ".csv"
    dataframe = pandas.read_csv(filename, header=0)
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:, :])
    return HEADER, DATA


def build_ann_sub(
    regression,
    input_dim=1000,
    output_dim=1,
    hidden_layers=1,
    mid_layer=49,
    neuro_non_trainable=100,
    nb_layer_untrain=1,
    neurons_per_layer=1000,
    activation_function="relu",
):
    """
    Build an ANN model with an hidden layer with non trainable weights at the end.
    Used for substitution study.
    """
    model = Sequential()

    # Adding the input layer
    model.add(
        Dense(neurons_per_layer, input_dim=input_dim, activation=activation_function)
    )

    # Adding hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation_function))

    model.add(
        Dense(mid_layer, input_dim=neurons_per_layer, activation=activation_function)
    )

    # Adding the non-trainable intermediate layer
    for i in range(nb_layer_untrain):
        model.add(NonTrainableLayer(output_dim=neuro_non_trainable))
    model.add(NonTrainableLayer(output_dim=1))

    # Adding the output layer
    if regression:
        model.add(Dense(output_dim, activation="linear"))
    else:
        model.add(Dense(output_dim, activation="sigmoid"))
    # Adjust activation based on the problem
    # (e.g., 'sigmoid' for binary classification)

    return model


def build_ann(
    regression,
    input_dim=1000,
    output_dim=1,
    hidden_layers=1,
    mid_layer=18,
    neurons_per_layer=1000,
    activation_function="relu",
):

    model = Sequential()

    # Adding the input layer
    model.add(
        Dense(neurons_per_layer, input_dim=input_dim, activation=activation_function)
    )

    # Adding hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation_function))

    model.add(
        Dense(mid_layer, input_dim=neurons_per_layer, activation=activation_function)
    )

    model.add(NonTrainableLayer(output_dim=1))
    # Adding the non-trainable intermediate layer

    # Adding the output layer
    if regression:
        model.add(Dense(output_dim, activation="linear"))
    else:
        model.add(Dense(output_dim, activation="sigmoid"))
    # Adjust activation based on the problem
    # (e.g., 'sigmoid' for binary classification)

    return model


def fit_model(
    tf_model,
    X_train,
    y_train,
    optimizer="adam",
    loss="mse",
    metrics=["mse"],
    epochs=100,
    batch=30,
    learning_rate=0.001,
):
    """Compile and fit"""
    # Compile the model with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    tf_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    tf_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch,
        validation_split=0.1,
        verbose=0,
    )
    return tf_model


def evaluate_model(tf_model, X_test, y_test):
    loss, q2 = tf_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}, Test R2: {q2:.4f}")

    # Make predictions on the test set
    y_pred = tf_model.predict(X_test)
    return y_pred, loss, q2


def get_data(trainingfile, normalize=False):
    X, Y = read_XY(trainingfile)
    # Standardize the data
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X, Y


def ann_substitution(
    filename,
    regression,
    untrainable_size,
    nb_layer_untrain=1,
    cross_val=5,
    n_hidden_dim=200,
    mid_layer=49,
    untrain_seed=0,
):
    metric_1_list = []  # Q2 for regression Acc for classification
    metric_2_list = []  # Q2 variance weighted for regression AUC for classification
    folder = "Regressions" if regression else "Classifications"

    X, Y = get_data(folder + "/paul_" + filename + ".csv")
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
        ann_model = build_ann_sub(
            input_dim=input_dimension,
            regression=regression,
            mid_layer=mid_layer,
            neuro_non_trainable=untrainable_size,
            nb_layer_untrain=nb_layer_untrain,
            output_dim=output_dimension,
            neurons_per_layer=n_hidden_dim,
        )

        # Display the model summary
        # ann_model.summary()

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


def ann_comparaison(
    filename, regression, cross_val=5, n_hidden_dim=1000, mid_layer=49, untrain_seed=0
):
    metric_1_list = []  # Q2 for regression Acc for classification
    metric_2_list = []  # Q2 variance weighted for regression AUC for classification
    folder = "Regressions" if regression else "Classifications"

    X, Y = get_data(folder + "/paul_" + filename + ".csv")
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

        # Display the model summary
        # ann_model.summary()

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


def mult_ds_ANN(list_ds, regression):
    summary_df = pandas.DataFrame()
    for ds in list_ds:
        metric_1, metric_2 = ann_comparaison(ds, regression, cross_val=5)
        name_ds = [ds for i in metric_1]
        ds_df = pandas.DataFrame([metric_1, metric_2, name_ds]).T
        summary_df = pandas.concat([summary_df, ds_df])
    if regression:
        summary_df.to_csv("IJCAI/Reg/ANN_comparison.csv")
    else:
        summary_df.to_csv("IJCAI/Clas/ANN_comparison.csv")


regression_small_list = [
    "cpu_activity",
    "concrete_compressive_strength",
    "energy_efficiency",
]
classification_small_list = ["diabetes", "wdbc", "phoneme"]
# mult_ds_ANN(classification_small_list, regression=False)


class NonTrainableLayer(Layer):
    """Class of layer with non-trainable weights"""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NonTrainableLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        init = tf.initializers.random_uniform(minval=-1.0, maxval=1.0)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[1], self.output_dim),
            initializer=init,
            trainable=False,
        )
        super(NonTrainableLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
