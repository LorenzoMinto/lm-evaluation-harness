import logging
import json
import time
from tqdm import tqdm
from requests.exceptions import RequestException

from mlc_chat import ChatModule, ChatConfig
from lm_eval.base import BaseLM

logger = logging.getLogger(__name__)


def get_result(response):
    # return response["token_logprobs"], response["is_greedy"]
    return response["logprob"], response["is_greedy"]


class MLCLM(BaseLM):
    def __init__(self, model_path, truncate=False):
        super().__init__()
        self.model_path = model_path
        self.truncate = truncate
        self.temperature = 0.0
        self.max_length = 1024
        chat_config = ChatConfig(max_gen_len=128, temperature=0.0)
        self.chat_module = ChatModule(model=model_path, chat_config=chat_config)

    # TODO(LorenzoMnto): fix this function, generate is currently broken
    def mlc_completion(self, context, continuation=None, stop=None, retries=3, delay=5, **kwargs):
        raise NotImplementedError()

    def loglikelihood(self, requests):
        if not requests:
            return []
        res = []
        for context, continuation in tqdm(requests):
            response = json.loads(self.chat_module._loglikelihood(context=context, continuation=continuation))
            # print(context, "|", continuation, response)
            # print()
            if response and "token_logprobs" in response and response["token_logprobs"]:
                logprob, is_greedy = get_result(response)
                res.append((logprob, is_greedy))
            else:
                logger.warning("Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list.")
        return res

    def greedy_until(self, requests):
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()

    def _model_call(self, inps):
        # Placeholder implementation
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Placeholder implementation
        raise NotImplementedError()

    def tok_encode(self, string: str):
        raise NotImplementedError()

    def tok_decode(self, tokens):
        raise NotImplementedError()

    @property
    def batch_size(self):
        # Placeholder implementation
        raise NotImplementedError()

    @property
    def device(self):
        # Placeholder implementation
        raise NotImplementedError()

    @property
    def eot_token_id(self):
        # Placeholder implementation
        raise NotImplementedError()

    def max_length(self):
        return self.max_length

    @property
    def max_gen_toks(self):
        # Placeholder implementation
        raise NotImplementedError()
