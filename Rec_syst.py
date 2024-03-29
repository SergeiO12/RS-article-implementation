class MovieLensRecommendationSystem(Task):
    def __init__(self, file='movielens.csv'):
        super().__init__()
        self.data = pd.read_csv(file)
        self.user_item_matrix = self.data.pivot(index='userId', columns='title', values=['rating', 'genre']).fillna(0)
        self.steps = 2
        self.stops = ['\nRecommended Movies:\n', None]

    def get_user_similarity(self, user_id):
        user_ratings_genres = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
        similarity_matrix = cosine_similarity(self.user_item_matrix, user_ratings_genres)
        sim_scores = pd.Series(similarity_matrix.flatten(), index=self.user_item_matrix.index)
        sim_scores = sim_scores.sort_values(ascending=False)
        return sim_scores

    def recommend_movies(self, user_id, top_n=5):
        sim_scores = self.get_user_similarity(user_id)
        similar_users = sim_scores.index[1:]  # Exclude the user itself
        recommended_movies = []
        for user in similar_users:
            movies_rated_genres = self.user_item_matrix.loc[user]
            unrated_movies = movies_rated_genres[movies_rated_genres['rating'] == 0].index
            recommended_movies.extend(unrated_movies)
            if len(recommended_movies) >= top_n:
                break
        return recommended_movies[:top_n]