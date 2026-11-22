#Purpose: Determine whether adding identity-related features improves fraud detection compared with Experiment A.

from src.experiment import run_experiment

# ============================================================
# EXPERIMENT B: TRANSACTION + IDENTITY FEATURES
# ============================================================

results = run_experiment("B")


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nExperiment B — Transaction + Identity Features")

print("\nMetrics:")
for metric, value in results["metrics"].items():
    print(f"{metric}: {value:.6f}")

print("\nConfusion Matrix:")
print(results["confusion_matrix"])


# ============================================================
# VALIDATION
# ============================================================

assert results["experiment"] == "B"
assert results["model"] is not None
assert results["preprocessor"] is not None
assert results["metrics"] is not None
assert results["confusion_matrix"].shape == (2, 2)

print("\nExperiment B completed successfully.")