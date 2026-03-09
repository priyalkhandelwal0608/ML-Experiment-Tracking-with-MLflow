import pandas as pd
from scipy.stats import ks_2samp

THRESHOLD = 0.05

def detect_drift(reference, production):

    drift_report = {}

    for column in reference.columns:

        stat, p_value = ks_2samp(
            reference[column],
            production[column]
        )

        drift_report[column] = {
            "p_value": p_value,
            "drift_detected": p_value < THRESHOLD
        }

    return drift_report