import pandas as pd
from base import Task
from pr.text import *
from models import gpt
import re
import random

class MovieLensRecommendationTask(Task):
    """
    Input (x)   : a user or item ID
    Output (y)  : recommended movies with average rating > 4
    Reward (r)  : # Define how reward is calculated for movie recommendations
    Input Example: 
    Output Example: 
    """
    def __init__(self, file='movielens.csv'):
        """
        file: CSV file containing MovieLens dataset information
        """
        super().__init__()
        path = file
        self.data = pd.read_csv(path)
        self.data = self.data.groupby('title').filter(lambda x: x['rating'].mean() > 4)  # Filter movies with avg rating > 4
        self.steps = 2
        self.stops = ['\nRecommended Movies:\n', None]

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return str(self.data.iloc[idx])
    
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
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = vote_prompt
        for i, y in enumerate(ys, 1):
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
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

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Recommended Movies:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Movie Recommendation 1:\n{ys[0]}\n\nMovie Recommendation 2:\n{ys[1]}\n'
        return prompt
    
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