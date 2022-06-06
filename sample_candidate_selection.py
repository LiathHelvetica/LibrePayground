from pyspark.sql import SparkSession
from properties import MASTER_URL, APP_NAME, MEMORY_IN_G, HDFS_URL_FORMAT, INTERACTIONS_FILE_PATH, HDFS_USER_ID, \
	USER_IDS_OUTPUT_PATH, USER_IDS_OUTPUT_FORMAT, MODEL_COLUMNS, MODEL_RANK, MODEL_MAX_ITER, MODEL_REG_PARAM, MODEL_PATH
from json import loads
from pyspark.ml.recommendation import ALSModel

SAMPLE_USER_ID = 2123
N_RECOMMENDATIONS = 1000

spark = SparkSession.builder\
	.master(MASTER_URL)\
	.config("spark.driver.memory", f"{MEMORY_IN_G}g")\
	.appName(APP_NAME)\
	.getOrCreate()

sample_user_df = spark.createDataFrame([tuple([SAMPLE_USER_ID])], [HDFS_USER_ID])

model = ALSModel.load(HDFS_URL_FORMAT.format(MODEL_PATH))

results = model.recommendForUserSubset(sample_user_df, numItems=N_RECOMMENDATIONS)
print(results)
