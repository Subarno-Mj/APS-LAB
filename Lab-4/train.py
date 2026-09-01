# ============================================================
# train.py
# ============================================================

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class BreastCancerTrain:

    # ========================================================
    # STEP 1: INITIALIZATION
    # ========================================================

    def __init__(self):

        self.model = None

        self.probabilities = None

        self.results = None

        self.threshold_metrics = None


    # ========================================================
    # STEP 2: CREATE LOGISTIC REGRESSION MODEL
    # ========================================================

    def create_model(self):

        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        )

        print(
            "\nLogistic Regression model created."
        )

        return self.model


    # ========================================================
    # STEP 3: TRAIN MODEL
    # ========================================================

    def train_model(
        self,
        X_train,
        y_train
    ):

        self.model.fit(
            X_train,
            y_train
        )

        print(
            "\nModel training completed."
        )

        return self.model


    # ========================================================
    # STEP 4: OBTAIN PREDICTED PROBABILITIES
    # ========================================================

    def get_predicted_probabilities(
        self,
        X_test
    ):

        self.probabilities = (
            self.model.predict_proba(X_test)
        )

        print(
            "\nPredicted probabilities:"
        )

        print(
            self.probabilities[:5]
        )

        return self.probabilities


    # ========================================================
    # STEP 5: CREATE RESULTS DATAFRAME
    # ========================================================

    def create_results_dataframe(
        self,
        y_test
    ):

        # model.classes_ = [0, 1]
        #
        # 0 = Benign
        # 1 = Malignant
        #
        # probabilities[:, 0] = Benign
        # probabilities[:, 1] = Malignant

        self.results = pd.DataFrame({

            "Actual_class":
                y_test.values,

            "p_benign":
                self.probabilities[:, 0],

            "p_malignant":
                self.probabilities[:, 1]
        })

        print("\nResults:")

        print(
            self.results.head(10)
        )

        return self.results


    # ========================================================
    # STEP 6: ADD ACTUAL LABEL
    # ========================================================

    def add_actual_label(self):

        self.results["Actual_label"] = (
            self.results["Actual_class"]
            .map({
                0: "Benign",
                1: "Malignant"
            })
        )

        print(
            self.results[
                [
                    "Actual_label",
                    "p_benign",
                    "p_malignant"
                ]
            ].head(10)
        )

        return self.results


    # ========================================================
    # STEP 7: GENERATE PREDICTIONS
    # ========================================================

    def generate_predictions(
        self,
        threshold=0.50
    ):

        self.results["Predicted_malignant"] = (
            self.results["p_malignant"]
            >= threshold
        ).astype(int)

        self.results["Predicted label"] = (
            self.results["Predicted_malignant"]
            .map({
                1: "Malignant",
                0: "Benign"
            })
        )

        print(
            f"\nPredictions using "
            f"threshold = {threshold}"
        )

        print(
            self.results[
                [
                    "Actual_label",
                    "p_malignant",
                    "Predicted label"
                ]
            ].head(10)
        )

        return self.results


    # ========================================================
    # STEP 8: COMPARE DIFFERENT THRESHOLDS
    # ========================================================

    def compare_thresholds(
        self,
        thresholds=[0.3, 0.5, 0.7]
    ):

        print(
            "\nPredicted malignant cases:"
        )

        for threshold in thresholds:

            predictions = (
                self.results["p_malignant"]
                >= threshold
            ).astype(int)

            print(
                f"Threshold : {threshold} ",
                f"Predicted malignant cases = "
                f"{predictions.sum()}"
            )


    # ========================================================
    # STEP 9: CONSTRUCT CONFUSION MATRIX
    # ========================================================

    def construct_confusion_matrix(
        self,
        y_test,
        thresholds=[0.3, 0.5, 0.7]
    ):

        # 0 = Benign
        # 1 = Malignant

        actual_malignant = y_test.values

        for threshold in thresholds:

            predicted_malignant = (
                self.results["p_malignant"]
                >= threshold
            ).astype(int)

            cm = confusion_matrix(
                actual_malignant,
                predicted_malignant
            )

            print(
                f"\nThreshold = {threshold}"
            )

            print(cm)


    # ========================================================
    # STEP 10: SKLEARN METRICS
    # ========================================================

    def calculate_metrics(
        self,
        y_test,
        threshold=0.50
    ):

        y_pred = (
            self.results["p_malignant"]
            >= threshold
        ).astype(int)

        print(
            f"\nSklearn metrics "
            f"at threshold = {threshold}"
        )

        print(
            "Sklearn accuracy:",
            accuracy_score(
                y_test,
                y_pred
            )
        )

        print(
            "Sklearn precision:",
            precision_score(
                y_test,
                y_pred
            )
        )

        print(
            "Sklearn recall:",
            recall_score(
                y_test,
                y_pred
            )
        )

        print(
            "Sklearn f1 score:",
            f1_score(
                y_test,
                y_pred
            )
        )

        return y_pred


    # ========================================================
    # STEP 11: THRESHOLD METRICS ANALYSIS
    # ========================================================

    def threshold_metrics_analysis(
        self,
        thresholds=[
            0.1,
            0.3,
            0.5,
            0.7,
            0.9
        ]
    ):

        threshold_metrics = []

        for threshold in thresholds:

            # Generate predictions

            y_pred = (
                self.results["p_malignant"]
                >= threshold
            ).astype(int)

            y_true = (
                self.results["Actual_class"]
            )


            # Confusion matrix

            tn, fp, fn, tp = (
                confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1]
                ).ravel()
            )


            # Metrics

            threshold_metrics.append({

                "Threshold":
                    threshold,

                "Accuracy":
                    accuracy_score(
                        y_true,
                        y_pred
                    ),

                "Precision":
                    precision_score(
                        y_true,
                        y_pred
                    ),

                "Recall":
                    recall_score(
                        y_true,
                        y_pred
                    ),

                "F1 Score":
                    f1_score(
                        y_true,
                        y_pred
                    ),

                "TN": tn,

                "FP": fp,

                "FN": fn,

                "TP": tp
            })


        self.threshold_metrics = (
            pd.DataFrame(
                threshold_metrics
            )
        )

        return self.threshold_metrics
