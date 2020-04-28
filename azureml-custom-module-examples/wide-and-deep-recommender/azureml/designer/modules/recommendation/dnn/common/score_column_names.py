# common constant defined for three kinds of tasks
USER_COLUMN = "User"
ITEM_COLUMN = "Item"

# types defined for score column names attr in DataFrameSchema
RECOMMENDATION_USER_COLUMN_TYPE = "Recommendation User Column"
RECOMMENDATION_ITEM_COLUMN_TYPE = "Recommendation Item Column"

# part for regression
SCORED_RATING = "Scored Rating"
TRUE_RATING = "Rating"

# types defined for score column names attr in DataFrameSchema
RECOMMENDATION_REGRESSION_SCORED_RATING_TYPE = "Recommendation Regression Scored Rating"


def build_regression_column_names():
    return [USER_COLUMN, ITEM_COLUMN, SCORED_RATING]


# for DataFrameSchema score column name attr support
def build_regression_column_name_keys():
    return [RECOMMENDATION_USER_COLUMN_TYPE, RECOMMENDATION_ITEM_COLUMN_TYPE,
            RECOMMENDATION_REGRESSION_SCORED_RATING_TYPE]


# part for ranking
# constants defined for recommending items from rated items
RATED_ITEM = "Rated Item"

# types defined for score column names attr in DataFrameSchema
RECOMMENDATION_RATED_RANKING_RATED_ITEM_TYPE = "Recommendation Rated Ranking Rated Item"


def build_rated_ranking_column_names(top_k: int):
    column_names = [USER_COLUMN]
    column_names += [f"{RATED_ITEM} {i}" for i in range(1, top_k + 1)]
    return column_names


# for DataFrameSchema score column name attr support
def build_rated_ranking_column_name_keys(top_k: int):
    column_name_keys = [RECOMMENDATION_USER_COLUMN_TYPE]
    column_name_keys += [f"{RECOMMENDATION_RATED_RANKING_RATED_ITEM_TYPE} {i}" for i in range(1, top_k + 1)]
    return column_name_keys


# constant defined for recommending items from all items/from unrated items
RECOMMENDED_ITEM = "Recommended Item"

# types defined for score column names attr in DataFrameSchema
RECOMMENDATION_RECOMMENDED_ITEM_TYPE = "Recommendation Recommended Item"


def build_ranking_column_names(top_k: int):
    column_names = [USER_COLUMN]
    column_names += [f"{RECOMMENDED_ITEM} {i}" for i in range(1, top_k + 1)]
    return column_names


# for DataFrameSchema score column name attr support
def build_ranking_column_name_keys(top_k: int):
    column_name_keys = [RECOMMENDATION_USER_COLUMN_TYPE]
    column_name_keys += [f"{RECOMMENDATION_RECOMMENDED_ITEM_TYPE} {i}" for i in range(1, top_k + 1)]
    return column_name_keys
