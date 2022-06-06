from pyspark.sql import SparkSession
from properties import MASTER_URL, APP_NAME, MEMORY_IN_G, HDFS_URL_FORMAT, INTERACTIONS_FILE_PATH, HDFS_USER_ID, \
	USER_IDS_OUTPUT_PATH, USER_IDS_OUTPUT_FORMAT, MODEL_COLUMNS, \
	MODEL_PATH,	RANKS, MAX_ITERATIONS, REGULARISATION_PARAMETERS, COLD_STRATEGIES, N_FOLDS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from json import loads
from pyspark.ml.recommendation import ALS


def create_data_extraction_function():
	def outcome(j):
		j = loads(j["value"])
		if not j["is_read"]:
			return []
		else:
			return [(
				j[MODEL_COLUMNS[0]],
				int(j[MODEL_COLUMNS[1]]),
				float(j[MODEL_COLUMNS[2]])
			)]
	return outcome


spark = SparkSession.builder\
	.master(MASTER_URL)\
	.config("spark.driver.memory", f"{MEMORY_IN_G}g")\
	.appName(APP_NAME)\
	.getOrCreate()

user_ids_df = spark.read\
	.option("header", "true")\
	.option("inferSchema", "true")\
	.format(USER_IDS_OUTPUT_FORMAT)\
	.load(HDFS_URL_FORMAT.format(USER_IDS_OUTPUT_PATH))

interactions_file = spark.read.text(HDFS_URL_FORMAT.format(INTERACTIONS_FILE_PATH))

model_data = interactions_file\
	.rdd\
	.flatMap(create_data_extraction_function())\
	.toDF(MODEL_COLUMNS)

model_data = model_data\
	.join(user_ids_df, model_data.user_id == user_ids_df.user_id, "inner")\
	.select(user_ids_df.hdfs_user_id, model_data.book_id, model_data.rating)

evaluator = RegressionEvaluator(metricName='rmse', labelCol=MODEL_COLUMNS[1], predictionCol=MODEL_COLUMNS[2])
als = ALS(
	userCol=HDFS_USER_ID,
	itemCol=MODEL_COLUMNS[1],
	ratingCol=MODEL_COLUMNS[2]
)
parameter_grid = ParamGridBuilder()\
	.addGrid(als.regParam, REGULARISATION_PARAMETERS)\
	.addGrid(als.rank, RANKS)\
	.addGrid(als.maxIter, MAX_ITERATIONS)\
	.addGrid(als.coldStartStrategy, COLD_STRATEGIES)\
	.build()

cross_validator = CrossValidator(
	estimator=als,
	estimatorParamMaps=parameter_grid,
	evaluator=evaluator,
	numFolds=N_FOLDS
)\
	.fit(model_data)

model = cross_validator.bestModel

regularisation_parameter = model._java_obj.parent().getRegParam()
rank = model.rank
max_iterations = model._java_obj.parent().getMaxIter()
cold_start_strategy = model._java_obj.parent().getColdStartStrategy()

als = ALS(
	rank=rank,
	maxIter=max_iterations,
	regParam=regularisation_parameter,
	userCol=HDFS_USER_ID,
	itemCol=MODEL_COLUMNS[1],
	ratingCol=MODEL_COLUMNS[2],
	coldStartStrategy=cold_start_strategy
)

model = als.fit(model_data)

model.write().overwrite().save(
	HDFS_URL_FORMAT.format(MODEL_PATH)
)
