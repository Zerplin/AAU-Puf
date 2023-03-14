import pypuf.simulation, pypuf.io, pypuf.attack, pypuf.metrics
import numpy as np
import pandas as pd
puf = pypuf.simulation.XORArbiterPUF(n=64, k=4, seed=1337)

pypuf.metrics.reliability(puf, seed=1337).mean()
challenges = pypuf.io.random_inputs(n=puf.challenge_length, N=500, seed=1337)
responses_mean = puf.r_eval(5, challenges).mean(axis=-1)
crps = pypuf.io.ChallengeResponseSet(challenges, responses_mean)

responses = crps.responses
challenge_response_pairs = []

for idx, response in np.ndenumerate(responses):
    challenge_response_row = np.append(crps.challenges[idx[0]], response)
    challenge_response_pairs.append(challenge_response_row)
challenge_response_pairs_df = pd.DataFrame(np.row_stack(challenge_response_pairs))
print(challenge_response_pairs_df)

challenge_response_pairs_df.to_csv('challenge_response.csv', index = False)
