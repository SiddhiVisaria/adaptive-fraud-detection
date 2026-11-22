from src.experiment import run_experiment


# Run Experiment A: transaction features only
results = run_experiment("A")


# Display results
print("\nExperiment A Results:")
print(results["metrics"])

print("\nConfusion Matrix:")
print(results["confusion_matrix"])


# Basic validation checks
assert results["experiment"] == "A"
assert results["model"] is not None
assert results["preprocessor"] is not None
assert results["metrics"] is not None
assert results["confusion_matrix"].shape == (2, 2)

print("\nAll experiment pipeline checks passed.")