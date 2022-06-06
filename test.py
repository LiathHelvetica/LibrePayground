from json import loads
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from numpy import matrix, asarray, zeros

APP_NAME = "TEST"

columns = ["user_id", "book_id", "rating"]
ratings = []
# user_id (string) to system_id (int)
users = {}
# book_id set (set of ints)
books = set()
user_id = 0

with open("assets/goodreads_interactions_comics_graphic.json") as f:
	line = f.readline()
	while line:
		data = loads(line)
		if data["is_read"]:
			current_user_id = 0
			if data["user_id"] in users:
				current_user_id = users[data["user_id"]]
			else:
				current_user_id = user_id
				users[data["user_id"]] = user_id
				user_id += 1
			book_id = int(data["book_id"])
			books.add(book_id)
			ratings.append((current_user_id, book_id, data["rating"]))
		line = f.readline()

"""""
with open("assets/goodreads_reviews_comics_graphic.json") as f:
	line = f.readline()
	while line:
		data = loads(line)
		ratings.append((data["user_id"], data["book_id"], data["rating"]))
		line = f.readline()
"""

spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "30g").appName(APP_NAME).getOrCreate()
df = spark.createDataFrame(ratings, columns)

RANK = 10
MAX_ITER = 10
REG_PARAM = 1.0

(training, test) = df.randomSplit([0.8, 0.2])
als = ALS(maxIter=MAX_ITER, regParam=REG_PARAM, userCol="user_id", itemCol="book_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(training)

# TODO: cache
item_factors = matrix(asarray(model.itemFactors.select("features").collect()))
item_factors_t = item_factors.T
#############

sample_user_book_history = zeros(model.itemFactors.count())
i = 0
for row in model.itemFactors.collect():
	if row["id"] % 500 == 0:
		sample_user_book_history[i] = 5.0
	i += 1

# fold-in approach - alternatively look for similar users
# vector of predictions
sample_user_prediction_result = sample_user_book_history @ item_factors @ item_factors_t

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
