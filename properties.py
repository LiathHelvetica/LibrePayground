"""
APP_NAME = "Application"
MASTER_URL = "spark://HOST:7077"
MEMORY_IN_G = 30

HDFS_URL_FORMAT = "hdfs://HOST:8020/{0}"
INTERACTIONS_FILE_PATH = "goodreads_interactions_comics_graphic.json"
BOOKS_FILE_PATH = "goodreads_books_comics_graphic.json"
USER_IDS_OUTPUT_PATH = "user_ids_map"
USER_IDS_OUTPUT_FORMAT = "csv"
USER_IDS_OUTPUT_MODE = "overwrite"

GOODREADS_USER_ID_KEY = "user_id"
HDFS_USER_ID = "user_id"

POSITIONAL_BOOK_ID = "id"
HDFS_BOOK_ID = "hdfs_book_id"

MODEL_COLUMNS = ["user_id", "book_id", "rating"]

MODEL_RANK = 10
MODEL_MAX_ITER = 10
MODEL_REG_PARAM = 1.0

MODEL_PATH = "assets/candidate_selector"
MODEL_OUTPUT_MODE = "overwrite"
"""

APP_NAME = "Application"
MASTER_URL = "local[*]"
MEMORY_IN_G = 30

HDFS_URL_FORMAT = "{0}"
INTERACTIONS_FILE_PATH = "assets/goodreads_interactions_comics_graphic.json"
BOOKS_FILE_PATH = "assets/goodreads_interactions_comics_graphic.json"
USER_IDS_OUTPUT_PATH = "assets/user_ids_map"
USER_IDS_OUTPUT_FORMAT = "csv"
USER_IDS_OUTPUT_MODE = "overwrite"

GOODREADS_USER_ID_KEY = "user_id"
HDFS_USER_ID = "hdfs_user_id"

POSITIONAL_BOOK_ID = "positional_id"
HDFS_BOOK_ID = "hdfs_book_id"

MODEL_COLUMNS = ["user_id", "book_id", "rating"]

MODEL_RANK = 10
MODEL_MAX_ITER = 10
MODEL_REG_PARAM = 1.0

MODEL_PATH = "assets/candidate_selector"
MODEL_OUTPUT_MODE = "overwrite"

# TODO: FIX BELOW PROPERTIES

RANKS = [10, 15]
MAX_ITERATIONS = [5, 10]
REGULARISATION_PARAMETERS = [0.01, 0.1]
COLD_STRATEGIES = ["nan", "drop"]
N_FOLDS = 3

CANDIDATE_SELECTION_REQUESTS_TOPIC = "candidate-selection-requests"
CANDIDATE_SELECTION_TOPIC = "candidate-selection"
CANDIDATE_SELECTION_CONSUMER_GROUP = "candidate-selection-nodes"
BOOTSTRAP_SERVERS = ["localhost:9092"]
KEY_REPLACEMENT = "NO_KEY"

USER_ID_KEY = "userId"
USER_HISTORY_KEY = "history"

BOOK_ID_KEY = "id"
BOOK_RATING_KEY = "rating"

PREDICTED_RATING_KEY = "predicted_rating"
PREDICTED_ID_KEY = "predicted_id"

N_CANDIDATES = 1000
