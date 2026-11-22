#Purpose: Determine whether explicitly telling the model whether identity information is available provides additional predictive value beyond the identity features themselves.

from src.experiment import run_experiment

# ============================================================
# EXPERIMENT C: TRANSACTION + IDENTITY + IDENTITY INDICATOR
# ============================================================

results = run_experiment("C")


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nExperiment C — Transaction + Identity + Availability Indicator")

print("\nMetrics:")
for metric, value in results["metrics"].items():
    print(f"{metric}: {value:.6f}")

print("\nConfusion Matrix:")
print(results["confusion_matrix"])


# ============================================================
# VALIDATION
# ============================================================

assert results["experiment"] == "C"
assert results["model"] is not None
assert results["preprocessor"] is not None
assert results["metrics"] is not None
assert results["confusion_matrix"].shape == (2, 2)

print("\nExperiment C completed successfully.")