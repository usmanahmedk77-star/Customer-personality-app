import pandas as pd
from src.clustering_pipeline import CustomerSegmentationModel

# Paths
DATA_PATH = r"data\customer_personality_feature_engineered.csv"
MODEL_PATH = r"models\clustering_model.joblib"
SEGMENTS_PATH = r"data\customer_segments.csv"

print("Loading feature-engineered dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# Build the exact production model defined in Module 6
print("Fitting CustomerSegmentationModel...")

model = CustomerSegmentationModel(
    k=4,
    random_state=42
)

model.fit(df)

# Generate predictions
df["Cluster"] = model.predict(df)
df["Segment_Name"] = model.predict_segment_names(df)

# Save the model
print("Saving production model...")
model.save(MODEL_PATH)

# Save segmented dataset
df.to_csv(SEGMENTS_PATH, index=False)

print("\nModel successfully rebuilt.")
print(f"Model: {MODEL_PATH}")
print(f"Segments: {SEGMENTS_PATH}")

print("\nSegment names:")
print(model.segment_names_)

print("\nCluster profile:")
print(model.get_cluster_profile())