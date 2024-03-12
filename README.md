# RecMind: Large Language Model Powered Agent For Recommendation    
## implementation using Tree of Thoughts algorithm

### RecMind consists of:

- Planning which enables the agent to break complex recommendation tasks into manageable steps for efficient handling of complex situations. Each step of planning involves thought, action and observation

- Memory consisting of Personalized Memory and World Knowledge, each accessible through specific tools.

- Tools enhance the agent functionality on top of the LLM, such as retrieving relevant knowledge, or assisting with reasoning process. 




![alt text](image-1.png?raw=true)


### Self-Inspiring.



### Self-Inspiring algorithm should break down a task to smaller sub-tasks for step-by-step planning.

### At each intermediate planning step, the agent “self-inspires” to consider all previously explored planning paths to explore the next planning states. Unlike existing Chain-of-Thoughts and Tree-of-Thoughts which discards states (thoughts) in previously explored paths when generating a new state, SI retains all previous states from all history paths when generating new state.


![alt text](image-2.png?raw=true)

![alt text](image-3.png)


### The Self_inspiring_DFS.py file includes the implementation of the Self-Inspiring algorithm based on the DFS algorithm made in the article Tree of Thoughts.

### Brief description of the code:

### the script includes functions for sampling from a distribution, incorporating a self-inspiring algorithm, building a tree of thoughts, and sampling the final response. The main algorithm is a while loop that samples the next step, checks for the end of planning, and either continues the current path or explores an alternative reasoning branch based on the "inspire" function.
### The "inspire" function initializes a tree, builds the tree of thoughts using depth-first search (DFS), and chooses the sequence with the highest score as the final sequence, as well the "inspire" function incorporates a self-inspiring algorithm that checks the cache for previously computed responses and expands the current node by adding child nodes for each candidate sequence in S. This suggests that the algorithm takes into account previous paths and uses them to inform the decision-making process for generating the next step of planning.

### The "sample_from_distribution" function samples responses and incorporates the self-inspiring algorithm by checking the cache for previously computed responses. 


### The code also includes a "compute_score" function for calculating the score of a sequence based on a provided scoring matrix and gap penalty.


## Functionality of all files:

### model.py


-	Setting OPENAI_KEY; 
-	calls chatgpt function to generate responses;
-	accumulates the generated responses and tracks the usage of completion tokens and prompt tokens; 
-	gpt_usage function calculates the estimated cost based on the accumulated completion tokens and prompt tokens for a specific GPT model.


### run.py

This script is designed to automate the execution of tasks with different configurations and methods, logging results for analysis and monitoring performance metrics.

### base.py

This file sets up a basic structure for a task-related class but lacks specific implementations for its methods.

### text.py

This file defines a class TextTask that inherits from a base class Task. The purpose of this class is to handle text generation tasks based on input text instructions. The file provides a structured approach to handling text generation tasks, including generating prompts, processing outputs, and evaluating coherency scores using a GPT model. It demonstrates a systematic way to manage text-based tasks within a larger project or system.

### __init__.py

This Python file defines a function called get_task(name) that takes a string parameter name and returns an instance of a specific task based on the value of name. 

### bfs.py

These functions collectively enable the user to interact with the GPT model to solve tasks efficiently by generating outputs, evaluating them based on historical information or voting mechanisms, and selecting the most promising candidates.

-	get_value(task, x, y, n_evaluate_sample, cache_value=True): This function retrieves the value for a given task, input x, and output candidate y. It utilizes a GPT model to generate outputs and caches values if specified.
-	get_values(task, x, ys, n_evaluate_sample, cache_value=True): This function obtains values for multiple output candidates ys by calling get_value() for each candidate while avoiding duplicates.
-	get_votes(task, x, ys, n_evaluate_sample): This function generates votes based on the input x and multiple output candidates ys using a GPT model.
-	get_proposals(task, x, y): This function generates proposals for a given task and input-output pair using a GPT model.
-	get_samples(task, x, y, n_generate_sample, prompt_sample, stop): This function generates samples based on the input-output pair (x, y) using different prompts like 'standard' or 'cot'.
-	def_solve(args, task, idx, to_print=True): This function solves a task by iteratively generating new output candidates based on different methods (sample or propose), evaluating them using historical information (vote or value), and selecting the best candidates (sample or greedy).
-	def_naive_solve(args, task, idx, to_print=True): This function provides a simpler approach to solving tasks by directly generating samples without the iterative selection process used in solve()






























links to RecMind article <https://arxiv.org/pdf/2308.14296v1.pdf>

link to Tree of Thoughts article <https://arxiv.org/pdf/2305.10601v2.pdf>