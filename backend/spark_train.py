from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# -------------------------------
# Start Spark Session
# -------------------------------
spark = SparkSession.builder \
    .appName("DiseasePrediction") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# -------------------------------
# Load Cleaned Data
# -------------------------------
df = spark.read.csv("data/cleaned_dataset.csv", header=True, inferSchema=True)

print("Columns:", df.columns)

target = "diseases"

# -------------------------------
# Drop null values
# -------------------------------
df = df.dropna()

# -------------------------------
# Label Encoding
# -------------------------------
indexer = StringIndexer(
    inputCol=target,
    outputCol="label",
    handleInvalid="keep"
)

# -------------------------------
# Feature Selection
# -------------------------------
features = [c for c in df.columns if c != target]

# Ensure all features are numeric
for c in features:
    df = df.withColumn(c, col(c).cast("double"))

assembler = VectorAssembler(
    inputCols=features,
    outputCol="features",
    handleInvalid="keep"
)

# -------------------------------
# Models
# -------------------------------
lr = LogisticRegression(
    labelCol="label",
    featuresCol="features",
    maxIter=100
)

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,
    maxDepth=10
)

# -------------------------------
# Pipelines
# -------------------------------
pipeline_lr = Pipeline(stages=[indexer, assembler, lr])
pipeline_rf = Pipeline(stages=[indexer, assembler, rf])

# -------------------------------
# Train-Test Split
# -------------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)

# -------------------------------
# Train Models
# -------------------------------
model_lr = pipeline_lr.fit(train)
model_rf = pipeline_rf.fit(train)

# -------------------------------
# Predictions
# -------------------------------
pred_lr = model_lr.transform(test)
pred_rf = model_rf.transform(test)

# -------------------------------
# Evaluation
# -------------------------------
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    metricName="accuracy"
)

acc_lr = evaluator.evaluate(pred_lr)
acc_rf = evaluator.evaluate(pred_rf)

print("Logistic Regression Accuracy:", acc_lr)
print("Random Forest Accuracy:", acc_rf)

# -------------------------------
# Select Best Model
# -------------------------------
if acc_rf > acc_lr:
    best_model_name = "Random Forest"
else:
    best_model_name = "Logistic Regression"

print("Best Model Selected:", best_model_name)

# -------------------------------
# Stop Spark
# -------------------------------
spark.stop()