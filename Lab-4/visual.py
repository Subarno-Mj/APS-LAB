# ============================================================
# visual.py
# ============================================================

import matplotlib.pyplot as plt

from IPython.display import display


class BreastCancerVisual:

    # ========================================================
    # STEP 1: PLOT CLASS DISTRIBUTION
    # ========================================================

    def plot_class_distribution(
        self,
        class_distribution
    ):

        class_distribution.plot(
            x="Class",
            y="Count",
            kind="bar",
            legend=False,
            color=["orange", "green"]
        )

        plt.ylabel(
            "Number of observations"
        )

        plt.xlabel(
            "Class"
        )

        plt.title(
            "Class Distribution"
        )

        plt.xticks(
            rotation=0
        )

        plt.tight_layout()

        plt.show()


    # ========================================================
    # STEP 2: DISPLAY THRESHOLD METRICS
    # ========================================================

    def display_threshold_metrics(
        self,
        threshold_metrics
    ):

        print(
            "\nThreshold metrics:"
        )

        display(
            threshold_metrics.round(3)
        )
