
# data.py


import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class BreastCancerData:

    # ========================================================
    # STEP 1: INITIALIZATION
    # ========================================================

    def __init__(self):

        self.data = None

        self.x = None
        self.y = None

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.class_distribution = None


    # ========================================================
    # STEP 2: LOAD THE DATASET
    # ========================================================

    def load_dataset(self):

        self.data = load_breast_cancer()

        print("Dataset loaded successfully.")

        return self.data


    # ========================================================
    # STEP 3: CREATE FEATURE MATRIX
    # ========================================================

    def create_feature_matrix(self):

        self.x = pd.DataFrame(
            self.data.data,
            columns=self.data.feature_names
        )

        print(
            "Feature matrix shape:",
            self.x.shape
        )

        return self.x


    # ========================================================
    # STEP 4: CREATE TARGET VARIABLE
    # ========================================================

    def create_target_variable(self):

        # Original sklearn target:
        #
        # 0 = Malignant
        # 1 = Benign
        #
        # Our target:
        #
        # 0 = Benign
        # 1 = Malignant

        self.y = pd.Series(
            (self.data.target == 0).astype(int),
            name="malignant"
        )

        print("\nTarget value counts:")
        print(self.y.value_counts())

        print(
            "Target shape:",
            self.y.shape
        )

        print(
            "Class names:",
            self.data.target_names
        )

        return self.y


    # ========================================================
    # STEP 5: EXAMINE CLASS DISTRIBUTION
    # ========================================================

    def examine_class_distribution(self):

        class_counts = (
            self.y
            .value_counts()
            .sort_index()
        )

        self.class_distribution = pd.DataFrame(
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

        print(self.class_distribution)

        return self.class_distribution


    # ========================================================
    # STEP 6: CREATE TRAINING AND TESTING SAMPLES
    # ========================================================

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
