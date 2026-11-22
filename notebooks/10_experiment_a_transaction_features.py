from src.experiment import run_experiment


# ============================================================
# EXPERIMENT A: TRANSACTION FEATURES ONLY
# ============================================================

results = run_experiment("A")


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nExperiment A — Transaction Features Only")

print("\nMetrics:")
for metric, value in results["metrics"].items():
    print(f"{metric}: {value:.6f}")

print("\nConfusion Matrix:")
print(results["confusion_matrix"])


# ============================================================
# VALIDATION
# ============================================================

assert results["experiment"] == "A"
assert results["model"] is not None
assert results["preprocessor"] is not None
assert results["metrics"] is not None

print("\nExperiment A completed successfully.")