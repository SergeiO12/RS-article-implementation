import pandas as pd
from base import Task
from pr.text import *
from langchain_experimental.agents import create_csv_agent
from langchain.agents import Tool

#Класс для рекомендации фильмов на основе данных из MovieLens.
class MovieLensRecommendationTask(Task):
    def __init__(self, file='movielens.csv'):
        #file: CSV-файл с данными из MovieLens.
        super().__init__()
        self.csv_agent = create_csv_agent(file)
        self.tools = [
            # Tool для поиска популярных фильмов в базе данных
            Tool(
                name="Find_Popular_Movies",
                func=self.find_popular_movies,
                description="Use this tool to find the most popular movies in the dataset"
            ),
            # Tool для поиска анимационных комедий в базе данных
            Tool(
                name="Find_Animated_Comedy",
                func=self.find_animated_comedy,
                description="Use this tool to find animated comedy movies in the dataset"
            ),
            # Tool для поиска малопопулярного фильма в базе данных
            Tool(
                name="Find_Less_Popular_Movie",
                func=self.find_less_popular_movie,
                description="Use this tool to find a less popular movie in the dataset"
            ),
            # Tool для поиска средней романтической драмы в базе данных
            Tool(
                name="Find_Average_Romantic_Drama",
                func=self.find_average_romantic_drama,
                description="Use this tool to find the average romantic drama in the movielens dataset"
            ),
            # Tool для рекомендации 5 комедий, снятых до 2000 года
            Tool(
                name="Recommend_Comedies_Before_2000",
                func=self.recommend_comedy_before_2000,
                description="Use this tool to recommend 5 comedies produced before 2000"
            )
        ]
        self.data = pd.read_csv(file)
        self.data = self.data.groupby('title').filter(lambda x: x['rating'].mean() > 4)  # Filter movies with avg rating > 4
        self.steps = 2
        self.stops = ['\nRecommended Movies:\n', None]
    # Возвращает количество фильмов в базе данных.
    def __len__(self) -> int:
        return len(self.data)
    # Возвращает строку с данными о фильме по его индексу.
    def get_input(self, idx: int) -> str:
        return str(self.data.iloc[idx])
    # Тестовая функция для оценки рекомендаций.
    def test_output(self, idx: int, output: str):
        output = output.split('Recommended Movies:\n')[-1]
        prompt = score_prompt + output
        score_outputs = gpt(prompt, n=5, model='gpt-3.5-turbo')
        scores = []
        for score_output in score_outputs:
            pattern = r".*relevance score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'------------------score no match: {[score_output]}')
        print(scores)
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
        return info
    # Форматирование стандартного промпта.
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y
    # Форматирование промпта для выбора комедии.
    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    # Форматирование промпта для голосования.
    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = vote_prompt
        for i, y in enumerate(ys, 1):
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    # Результаты голосования.
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results
    # Форматирование промпта для сравнения.
    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Recommended Movies:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Movie Recommendation 1:\n{ys[0]}\n\nMovie Recommendation 2:\n{ys[1]}\n'
        return prompt
    # Разбор результата сравнения.
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more relevant movie recommendation is 1' in compare_output:
            return 0
        elif 'more relevant movie recommendation is 2' in compare_output:
            return 1
        elif 'two movie recommendations are equally relevant' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1
    # Поиск популярных фильмов в базе данных movielens.
    def find_popular_movies(self, query: str) -> str:
        popular_movies = self.csv_agent.csv_to_table(query=query)
        return self.format_recommendations(popular_movies)
    # Поиск анимационных комедий в базе данных movielens
    def find_animated_comedy(self, query: str) -> str:
        animated_comedy = self.csv_agent.csv_to_table(query=query)
        return self.format_recommendations(animated_comedy)
    # Поиск малопопулярных фильмов в базе данных movielens.
    def find_less_popular_movie(self, query: str) -> str:
        less_popular_movie = self.csv_agent.csv_to_table(query=query).sample()
        return self.format_recommendations(less_popular_movie)
    # Поиск среднюю романтической драмы в базе данных.
    def find_average_romantic_drama(self, query: str) -> str:
        romantic_dramas = self.csv_agent.csv_to_table(query=query)
        avg_romantic_drama = romantic_dramas[romantic_dramas['genre'].str.contains('Romance|Drama')].mean()
        return avg_romantic_drama.to_string()
    # Рекомендация 5 комедий, снятых до 2000 года.
    def recommend_comedy_before_2000(self, query: str) -> str:
        comedies_before_2000 = self.csv_agent.csv_to_table(query=query)
        comedies_before_2000 = comedies_before_2000[comedies_before_2000['release_year'] < 2000 & comedies_before_2000['genre'].str.contains('Comedy')]
        recommendations = comedies_before_2000.sample(5)
        return self.format_recommendations(recommendations)
    # Сопоставление промптов с соответствующими инструментами.
    def map_prompt_to_tool(self, prompt: str) -> str:
        prompt_tool_mapping = {
            standard_prompt: "Find_Popular_Movies",
            cot_prompt: "Find_Animated_Comedy",
            vote_prompt: "Find_Less_Popular_Movie",
            compare_prompt: "Find_Average_Romantic_Drama",
            score_prompt: "Recommend_Comedies_Before_2000"
        }
        return prompt_tool_mapping.get(prompt, "No matching tool found")
    # Обрабатывает входной промпт и определяет, какой инструмент использовать.
    def process_input_prompt(self, prompt: str) -> str:
        tool_name = self.map_prompt_to_tool(prompt)
        if tool_name != "No matching tool found":
            return getattr(self, tool_name)()
        else:
            return "Could not find a tool for the given prompt."
    # Форматирование рекомендации фильмов.
    def format_recommendations(self, movies: pd.DataFrame) -> str:
        recommendations = movies.to_string(index=False)
        return f'\nRecommended Movies:\n{recommendations}'