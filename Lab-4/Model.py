# IMPORT LIBRARIES


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from IPython.display import display

# BREAST CANCER MODEL CLASS


class BreastCancerModel:
    
    # STEP 1: INITIALIZATION
    

    def __init__(self):

        self.data = None

        self.x = None
        self.y = None

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.model = None

        self.probabilities = None

        self.results = None

        self.threshold_metrics = None


    
    # STEP 2: LOAD THE DATASET
    

    def load_dataset(self):

        self.data = load_breast_cancer()

        print("Dataset loaded successfully.")

        return self.data


   
    # STEP 3: CREATE FEATURE MATRIX
    

    def create_feature_matrix(self):

        self.x = pd.DataFrame(
            self.data.data,
            columns=self.data.feature_names
        )

        print("Feature matrix shape:", self.x.shape)

        return self.x


    
    # STEP 4: CREATE TARGET VARIABLE
    

    def create_target_variable(self):

        # Original sklearn target:
        #
        # 0 = Malignant
        # 1 = Benign
        #
        # We convert it to:
        #
        # 0 = Benign
        # 1 = Malignant

        self.y = pd.Series(
            (self.data.target == 0).astype(int),
            name="malignant"
        )

        print("\nTarget value counts:")
        print(self.y.value_counts())

        print("Target shape:", self.y.shape)

        print("Class names:", self.data.target_names)

        return self.y


   
    # STEP 5: EXAMINE CLASS DISTRIBUTION
    

    def examine_class_distribution(self):

        class_counts = (
            self.y
            .value_counts()
            .sort_index()
        )

        class_distribution = pd.DataFrame(
            {
                "Class": [
                    "Benign",
                    "Malignant"
                ],

                "Count": class_counts.values,

                "Probability": (
                    class_counts.values /
                    len(self.y)
                )
            }
        )

        print("\nClass distribution:")

        print(class_distribution)

        return class_distribution


    
    # STEP 6: PLOT CLASS DISTRIBUTION
    

    def plot_class_distribution(self):

        class_counts = (
            self.y
            .value_counts()
            .sort_index()
        )

        class_distribution = pd.DataFrame(
            {
                "Class": [
                    "Benign",
                    "Malignant"
                ],

                "Count": class_counts.values
            }
        )

        class_distribution.plot(
            x="Class",
            y="Count",
            kind="bar",
            legend=False,
            color=["orange", "green"]
        )

        plt.ylabel("Number of observations")
        plt.xlabel("Class")
        plt.title("Class Distribution")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


    
    # STEP 7: CREATE TRAINING AND TESTING SAMPLES
    

    def create_train_test_samples(self):

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            self.x,
            self.y,
            test_size=0.2,
            random_state=42
        )

        print(
            "\nTraining size =",
            len(self.y_train)
        )

        print(
            "Testing size =",
            len(self.y_test)
        )

        print(
            "Training proportions ="
        )

        print(
            self.y_train
            .value_counts(normalize=True)
            .sort_index()
        )

        return (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        )


    
    # STEP 8: CREATE LOGISTIC REGRESSION MODEL
    

    def create_model(self):

        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        )

        print("\nLogistic Regression model created.")

        return self.model


    
    # STEP 9: TRAIN MODEL
    

    def train_model(self):

        self.model.fit(
            self.X_train,
            self.y_train
        )

        print("\nModel training completed.")

        return self.model


   
    # STEP 10: OBTAIN PREDICTED PROBABILITIES
    

    def get_predicted_probabilities(self):

        self.probabilities = (
            self.model.predict_proba(
                self.X_test
            )
        )

        print("\nPredicted probabilities:")

        print(
            self.probabilities[:5]
        )

        return self.probabilities


   
    # STEP 11: CREATE RESULTS DATAFRAME
    

    def create_results_dataframe(self):

        # model.classes_ = [0, 1]
        #
        # 0 = Benign
        # 1 = Malignant
        #
        # probabilities[:, 0] = Benign
        # probabilities[:, 1] = Malignant

        self.results = pd.DataFrame({

            "Actual_class":
                self.y_test.values,

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


    
    # STEP 12: ADD ACTUAL LABEL
    

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


    
    # STEP 13: GENERATE THRESHOLD PREDICTIONS
    

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


    
    # STEP 14: COMPARE DIFFERENT THRESHOLDS
    
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


    
    # STEP 15: CONSTRUCT CONFUSION MATRIX
    

    def construct_confusion_matrix(
        self,
        thresholds=[0.3, 0.5, 0.7]
    ):

        # 0 = Benign
        # 1 = Malignant

        actual_malignant = (
            self.y_test.values
        )

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


    
    # STEP 16: CALCULATE SKLEARN METRICS
    

    def calculate_metrics(
        self,
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
                self.y_test,
                y_pred
            )
        )

        print(
            "Sklearn precision:",
            precision_score(
                self.y_test,
                y_pred
            )
        )

        print(
            "Sklearn recall:",
            recall_score(
                self.y_test,
                y_pred
            )
        )

        print(
            "Sklearn f1 score:",
            f1_score(
                self.y_test,
                y_pred
            )
        )

        return y_pred


    
    # STEP 17: COMPARE METRICS FOR DIFFERENT THRESHOLDS
    

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


        # Convert to DataFrame

        self.threshold_metrics = (
            pd.DataFrame(
                threshold_metrics
            )
        )


        # Display results

        print(
            "\nThreshold metrics:"
        )

        display(
            self.threshold_metrics.round(3)
        )

        return self.threshold_metrics
