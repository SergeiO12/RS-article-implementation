def get_task(name):
    if name == 'rec_try':
        from rec_task import MovieLensRecommendationTask
        return MovieLensRecommendationTask()
    else:
        raise NotImplementedError
