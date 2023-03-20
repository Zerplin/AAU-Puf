# from pypuf.simulation import ArbiterPUF
import numpy as np
import pandas as pd
from pypuf.io import random_inputs
from pypuf.simulation import ArbiterPUF


class DatasetGenerator:
    """
    returns an arbiter puf
    """

    @staticmethod
    def _get_arbiter_puf(input_dim, seed):
        return ArbiterPUF(n=input_dim, seed=seed)

    """
    returns an array of challenges
    """

    @staticmethod
    def _get_challenges(input_dim, number_of_challenges, seed):
        return random_inputs(n=input_dim, N=number_of_challenges, seed=seed)

    @staticmethod
    def _get_challenge_response_pairs(puf, challenges):
        responses = puf.eval(challenges)
        challenge_response_pairs = []

        for idx, response in np.ndenumerate(responses):
            challenge_response_list = np.array(challenges[idx].tolist() + [response])
            challenge_response_pairs.append(challenge_response_list)
        challenge_response_pairs_df = pd.DataFrame(np.row_stack(challenge_response_pairs))
        return challenge_response_pairs_df

    """
    returns a pandas dataframe, last element is the response, rest is the challenge bits
    """

    @staticmethod
    def get_arbiter_dataset(input_dim=2, puf_seed=1337, challenge_seed=1337, number_of_challenges=3):
        puf = DatasetGenerator._get_arbiter_puf(input_dim, puf_seed)
        challenges = DatasetGenerator._get_challenges(input_dim, number_of_challenges, challenge_seed)
        return DatasetGenerator._get_challenge_response_pairs(puf, challenges)
