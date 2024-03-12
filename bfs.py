import itertools
import numpy as np
from functools import partial
from models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    explored_states = {}  # Dictionary to store explored states and their values for all paths

    for step in range(task.steps):
        new_explored_states = {}  # Temporary dictionary to store updated values for all paths

        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample,
                                  prompt_sample=args.prompt_sample,
                                  stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))

        # evaluation using historical information for all paths
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys,
                               args.n_evaluate_sample)
        # When args.method_evaluate is set to 'value', the code iterates 
        # over explored states and calculates path values for each candidate output y in new_ys.
        # If a candidate y has been previously explored and its value is stored in explored_states, 
        # that value is used. Otherwise, get_value function is called to determine its value based on historical data.
        # The calculated path values are then stored in new_explored_states for further reference.
        elif args.method_evaluate == 'value':
            for path in explored_states:
                path_values = [explored_states[path][y] if y in explored_states[path] else get_value(task, x, y, args.n_evaluate_sample) for y in new_ys]
                new_explored_states[path] = {y: path_values[i] for i, y in enumerate(new_ys)}

        # selection
                
        # The selection process relies on historical information to
        # make informed decisions on which candidates to choose for the next step.
        # When args.method_select is set to 'greedy', the code considers either 
        # the maximum value from explored states or current values depending on whether there are existing explored states.
        
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids,
                                          size=args.n_select_sample,
                                          p=ps).tolist()
        # If there are explored states available (checked with if step > 0 and explored_states), the code selects candidates 
        # based on their maximum historical values from these states. Otherwise, it selects candidates based on current values.
        elif args.method_select == 'greedy':
            if step > 0 and explored_states:  # Check if there are explored states
                select_ids = sorted(ids, key=lambda x: max([explored_states.get(path, {}).get(new_ys[x], 0) for path in explored_states]), reverse=True)[:args.n_select_sample]
            else:
                select_ids = sorted(ids,
                                    key=lambda x: values[x],
                                    reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # This selection mechanism ensures that past knowledge influences the decision-making process,
        # leading to potentially more optimal choices.

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})

        # Update explored_states with updated values for all paths
        if step > 0:
            for path in explored_states:
                explored_states[path].update(new_explored_states.get(path,{}))

        ys = select_new_ys

    if to_print:
        print(ys)

    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}

