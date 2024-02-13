

# The compute_score function calculates the score of a sequence based on a provided scoring matrix and gap penalty.
# It iterates through the sequence, adding the score based on the scoring matrix and gap penalty for gaps.
def compute_score(sequence, scoring_matrix, gap_penalty):
    score = 0
    for i in range(len(sequence)):
        if sequence[i] == '-':
            score += gap_penalty
        else:
            score += scoring_matrix[sequence[i]][i]
    return score

# The sample_from_distibution function samples responses and incorporates a self-inspiring algorithm by checking the cache for previously computed responses. 
# It uses the provided parameters to sample responses and update the cache with the computed scores.

def sample_from_distribution(x, S, theta):
    obs = x.render()
    if obs in x.cache: 
        print('cache hit')
        return x.cache[obs]
    print('call gpt')
    responses = gpt(prompt_wrap(obs), model='gpt-4', n=8)
    candidates_to_scores = {}
    for response in responses:
        parsed_response = parse_response(response)
        if parsed_response:
            for candidate, score in parsed_response:
                candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
    # Incorporate self-inspiring algorithm
    for s in S:
        state = x.apply_sequence(s)
        if state.is_terminal():
            continue
        obs = state.render()
        if obs in x.cache:
            candidates_to_scores.update(x.cache[obs])
        else:
            print('call gpt')
            responses = gpt(prompt_wrap(obs), model='gpt-4', n=8)
            for response in responses:
                parsed_response = parse_response(response)
                if parsed_response:
                    for candidate, score in parsed_response:
                        candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
            x.cache[obs] = candidates_to_scores.copy()
    x.cache[obs] = candidates_to_scores
    return candidates_to_scores

# The inspire function initializes a tree, builds the tree of thoughts using depth-first search (DFS),
# and chooses the sequence with the highest score as the final sequence. 
# It returns the final sequence.

def inspire(x, S):
    # Initialize the root node of the tree
    root = {'state': x, 'sequence': [], 'score': 0, 'children': []}
    # Build the tree of thoughts using DFS
    build_tree(root, S)
    # Choose the sequence with the highest score as the final sequence
    final_sequence = max(root['children'], key=lambda x: x['score'])['sequence']
    return final_sequence

# The build_tree is a helper function for the inspire function.
# It expands the current node by adding child nodes for each candidate sequence in S and computes the score of the current node as the maximum score of its children.

def build_tree(node, S):
    # Check if the current node is a leaf node
    if node['state'].is_terminal():
        node['score'] = compute_score(node['sequence'])
        return
    # Expand the current node by adding child nodes for each candidate sequence in S
    for s in S:
        child_node = {'state': node['state'].apply_sequence(s), 'sequence': node['sequence'] + [s], 'score': 0, 'children': []}
        build_tree(child_node, S)
        node['children'].append(child_node)
    # Compute the score of the current node as the maximum score of its children
    node['score'] = max(node['children'], key=lambda x: x['score'])['score']

# The sample_final_response function gets the sequence of actions to inspire the final response, 
# samples the final response from the distribution p_theta conditioned on the final sequence, and returns the final response.

def sample_final_response(x, S, theta):
    # Get the sequence of actions to inspire the final response
    sequence = inspire(x, S)
    # Sample the final response from the distribution p_theta conditioned on the final sequence
    final_response = sample_from_distribution(x, sequence, theta)
    return final_response