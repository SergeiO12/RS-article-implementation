standard_prompt = '''
Find a movie produced in USA with budget lower than 1 million dollars {input} 
'''

cot_prompt = '''
Recommend 5 movies similar to {input} but not from USA made in the period from 1996 to 1999
'''


vote_prompt = '''Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
'''

compare_prompt = '''Briefly analyze the coherency of the following two passages. Conclude in the last line "The more coherent passage is 1", "The more coherent passage is 2", or "The two passages are similarly coherent".
'''

score_prompt = '''Analyze the following passage, then at the last line conclude "Thus the coherency score is {s}", where s is an integer from 1 to 10.
'''