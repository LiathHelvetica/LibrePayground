from pyspark.sql import SparkSession
from properties import MASTER_URL, APP_NAME, MEMORY_IN_G, HDFS_URL_FORMAT, INTERACTIONS_FILE_PATH,\
	GOODREADS_USER_ID_KEY, HDFS_USER_ID, USER_IDS_OUTPUT_PATH, USER_IDS_OUTPUT_FORMAT, USER_IDS_OUTPUT_MODE
from json import loads

spark = SparkSession.builder\
	.master(MASTER_URL)\
	.config("spark.driver.memory", f"{MEMORY_IN_G}g")\
	.appName(APP_NAME)\
	.getOrCreate()

interactions_file = spark.read.text(HDFS_URL_FORMAT.format(INTERACTIONS_FILE_PATH))

user_ids_df = interactions_file\
	.rdd\
	.map(lambda j: loads(j["value"])[GOODREADS_USER_ID_KEY])\
	.distinct()\
	.zipWithIndex()\
	.toDF([GOODREADS_USER_ID_KEY, HDFS_USER_ID])

user_ids_df.write\
	.option("header", "true")\
	.save(
		HDFS_URL_FORMAT.format(USER_IDS_OUTPUT_PATH),
		format=USER_IDS_OUTPUT_FORMAT,
		mode=USER_IDS_OUTPUT_MODE
	)






























