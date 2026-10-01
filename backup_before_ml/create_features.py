import pandas as pd

# Read access log
df = pd.read_csv(
    "access_log.csv",
    names=["key", "operation", "timestamp"]
)

# Convert timestamp to numeric
df["timestamp"] = pd.to_numeric(df["timestamp"])

# Count operations for each key
features = df.groupby("key").agg(
    total_accesses=("key", "count"),
    get_count=("operation", lambda x: (x == "GET").sum()),
    set_count=("operation", lambda x: (x == "SET").sum()),
    delete_count=("operation", lambda x: (x == "DELETE").sum())
).reset_index()

# Save feature dataset
features.to_csv(
    "ml_features.csv",
    index=False
)

print(features)
print("\nFeature dataset created successfully!")