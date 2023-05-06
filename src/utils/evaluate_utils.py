# check the accuracy of the model
def get_accuracy(obs_sequence, most_likely_states):
    correct_predictions = 0
    for i in range(len(obs_sequence)):
        if obs_sequence[i] == most_likely_states[i]:
            correct_predictions += 1
    return correct_predictions / len(obs_sequence)