from pyspark.sql import SparkSession
from properties import MASTER_URL, APP_NAME, MEMORY_IN_G, HDFS_URL_FORMAT, INTERACTIONS_FILE_PATH, HDFS_USER_ID, \
	USER_IDS_OUTPUT_PATH, USER_IDS_OUTPUT_FORMAT, MODEL_COLUMNS, MODEL_RANK, MODEL_MAX_ITER, MODEL_REG_PARAM, MODEL_PATH
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
				j[MODEL_COLUMNS[2]]
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

als = ALS(
	rank=MODEL_RANK,
	maxIter=MODEL_MAX_ITER,
	regParam=MODEL_REG_PARAM,
	userCol=HDFS_USER_ID,
	itemCol=MODEL_COLUMNS[1],
	ratingCol=MODEL_COLUMNS[2],
	coldStartStrategy="drop"
)

model = als.fit(model_data)

model.save(
	HDFS_URL_FORMAT.format(MODEL_PATH)
)






























