from pyspark.sql import SparkSession
from kafka import KafkaConsumer, KafkaProducer
from properties import CANDIDATE_SELECTION_CONSUMER_GROUP, CANDIDATE_SELECTION_TOPIC, BOOTSTRAP_SERVERS,\
	CANDIDATE_SELECTION_REQUESTS_TOPIC, MASTER_URL, MEMORY_IN_G, APP_NAME, N_CANDIDATES, HDFS_URL_FORMAT, MODEL_PATH, \
	USER_HISTORY_KEY, USER_ID_KEY, HDFS_USER_ID, KEY_REPLACEMENT, POSITIONAL_BOOK_ID, HDFS_BOOK_ID, BOOK_ID_KEY, \
	BOOK_RATING_KEY, PREDICTED_RATING_KEY, PREDICTED_ID_KEY
from json import loads
from pyspark.ml.recommendation import ALSModel
from numpy import matrix, asarray, zeros


def deserialise_value(value):
	return loads(value.decode("utf-8"))


def deserialise_key(value):
	return value.decode("utf-8") if value is not None else KEY_REPLACEMENT


spark = SparkSession.builder\
	.master(MASTER_URL)\
	.config("spark.driver.memory", f"{MEMORY_IN_G}g")\
	.appName(APP_NAME)\
	.getOrCreate()

LOGGER = spark._jvm.org.apache.log4j.LogManager.getLogger(__name__)

cs_model = ALSModel.load(HDFS_URL_FORMAT.format(MODEL_PATH))

item_factors = matrix(asarray(cs_model.itemFactors.select("features").collect()))
item_factors_t = item_factors.T
n_books = cs_model.itemFactors.count()
books = cs_model.itemFactors.select("id").rdd.map(lambda r: r[0]).zipWithIndex().toDF([HDFS_BOOK_ID, POSITIONAL_BOOK_ID])

consumer = KafkaConsumer(
	CANDIDATE_SELECTION_REQUESTS_TOPIC,
	group_id=CANDIDATE_SELECTION_CONSUMER_GROUP,
	bootstrap_servers=BOOTSTRAP_SERVERS,
	value_deserializer=deserialise_value,
	key_deserializer=deserialise_key
)

producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS)

LOGGER.info("Candidate selection node - ready")
for message in consumer:
	user_id = message.value[USER_ID_KEY]
	history = message.value[USER_HISTORY_KEY]
	user_df = spark.createDataFrame([tuple([user_id])], [HDFS_USER_ID])
	results = cs_model.recommendForUserSubset(user_df, numItems=N_CANDIDATES)
	recommendations = results.select("recommendations").collect()
	if recommendations:
		recommendations_id_list = list(map(lambda r: r.book_id, recommendations[0][0]))
	else:
		user_data = zeros(n_books)
		user_history = list(map(lambda o: (o[BOOK_ID_KEY], o[BOOK_RATING_KEY]), history))
		user_history = spark.createDataFrame(user_history, [BOOK_ID_KEY, BOOK_RATING_KEY])
		user_history_positional = user_history.join(books, user_history[BOOK_ID_KEY] == books[HDFS_BOOK_ID], "inner")
		for row in user_history_positional.rdd.toLocalIterator():
			user_data[row[POSITIONAL_BOOK_ID]] = row[BOOK_RATING_KEY]
		user_prediction_result = (user_data @ item_factors @ item_factors_t).A[0, :].tolist()
		user_predictions_df = spark.createDataFrame(
			list(enumerate(user_prediction_result)), [PREDICTED_ID_KEY, PREDICTED_RATING_KEY]
		)
		user_predictions_df = user_predictions_df.join(
			books,
			books[POSITIONAL_BOOK_ID] == user_predictions_df[PREDICTED_ID_KEY],
			"inner"
		)
		recommendations_id_list = list(map(
			lambda r: r[HDFS_BOOK_ID],
			user_predictions_df.orderBy(user_predictions_df[PREDICTED_RATING_KEY].desc()).head(N_CANDIDATES)
		))
	# recommendations_id_list - list of goodreads books ids - selected candidates
	# TODO: candidate ranking
	print(recommendations_id_list)
