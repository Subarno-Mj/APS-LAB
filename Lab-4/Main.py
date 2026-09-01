# main - Driver program 
# data.py - the dataset
# visual.py - display results , graphs 
# train.py - traning model , prediction , metrics 

# And the execution flow is:
""" 
main.py
   │
   ├── data.load_dataset()
   ├── data.create_feature_matrix()
   ├── data.create_target_variable()
   ├── data.examine_class_distribution()
   │
   ├── visual.plot_class_distribution()
   │
   ├── data.create_train_test_samples()
   │
   ├── train.create_model()
   ├── train.train_model()
   ├── train.get_predicted_probabilities()
   ├── train.create_results_dataframe()
   ├── train.add_actual_label()
   ├── train.generate_predictions()
   ├── train.compare_thresholds()
   ├── train.construct_confusion_matrix()
   ├── train.calculate_metrics()
   ├── train.threshold_metrics_analysis()
   │
   └── visual.display_threshold_metrics()

"""


from data import BreastCancerData
from train import BreastCancerTrain
from visual import BreastCancerVisual


class Main:

    def run(self):

        # ====================================================
        # CREATE OBJECTS
        # ====================================================

        data = BreastCancerData()

        train = BreastCancerTrain()

        visual = BreastCancerVisual()


        # ====================================================
        # DATA OPERATIONS
        # ====================================================

        # Step 1: Load dataset

        data.load_dataset()


        # Step 2: Create feature matrix

        data.create_feature_matrix()


        # Step 3: Create target variable

        data.create_target_variable()


        # Step 4: Examine class distribution

        class_distribution = (
            data.examine_class_distribution()
        )


        # ====================================================
        # VISUALIZATION
        # ====================================================

        # Step 5: Plot class distribution

        visual.plot_class_distribution(
            class_distribution
        )


        # ====================================================
        # TRAIN / TEST DATA
        # ====================================================

        # Step 6: Create training and testing samples

        data.create_train_test_samples()


        # ====================================================
        # MODEL TRAINING
        # ====================================================

        # Step 7: Create model

        train.create_model()


        # Step 8: Train model

        train.train_model(
            data.X_train,
            data.y_train
        )


        # ====================================================
        # PREDICTIONS
        # ====================================================

        # Step 9: Get predicted probabilities

        train.get_predicted_probabilities(
            data.X_test
        )


        # Step 10: Create results dataframe

        train.create_results_dataframe(
            data.y_test
        )


        # Step 11: Add actual labels

        train.add_actual_label()


        # ====================================================
        # THRESHOLD = 0.50
        # ====================================================

        # Step 12: Generate predictions

        train.generate_predictions(
            threshold=0.50
        )


        # ====================================================
        # THRESHOLD COMPARISON
        # ====================================================

        # Step 13: Compare different thresholds

        train.compare_thresholds(
            thresholds=[
                0.3,
                0.5,
                0.7
            ]
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        # Step 14: Construct confusion matrix

        train.construct_confusion_matrix(
            data.y_test,
            thresholds=[
                0.3,
                0.5,
                0.7
            ]
        )


        # ====================================================
        # SKLEARN METRICS
        # ====================================================

        # Step 15: Calculate metrics

        train.calculate_metrics(
            data.y_test,
            threshold=0.50
        )


        # ====================================================
        # THRESHOLD METRICS
        # ====================================================

        # Step 16: Calculate metrics for
        # different thresholds

        threshold_metrics = (
            train.threshold_metrics_analysis(
                thresholds=[
                    0.1,
                    0.3,
                    0.5,
                    0.7,
                    0.9
                ]
            )
        )


        # ====================================================
        # DISPLAY THRESHOLD METRICS
        # ====================================================

        visual.display_threshold_metrics(
            threshold_metrics
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main = Main()

    main.run()

"""
cancer_prediction/
│
├── data.py
│
├── train.py
│
├── visual.py
│
└── main.py
"""