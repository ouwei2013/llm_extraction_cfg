
from cfg import CFG
import torch


class CFGModel:

    def __init__(self, model, tokenizer, cfg) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

    def generate(self, inputs, max_length=512):

        cur_txt = ''
        new_token = ''
        new_token_id = None
        past_key_values = None

        while len(cur_txt) < max_length:

            allowed_token_ids, allowed_tokens = self.cfg.get_accept_token_ids(
                cur_txt)

            if allowed_token_ids is None:
                break

            if past_key_values:
                inputs = self.model.prepare_inputs_for_generation(
                    new_token_id, past_key_values)
                inputs['use_cache'] = True
                output = self.model(**inputs)
            else:
                output = self.model(inputs, use_cache=True)

            past_key_values = output['past_key_values']
            scores = output['logits']
            scores = scores[:, -1, :]

            mask = torch.ones_like(scores) * -1e10

            for token_id in allowed_token_ids:
                mask[:, token_id] = 0

            scores = scores + mask
            new_token_id = torch.argmax(scores)
            new_token = self.tokenizer.decode(new_token_id)
            new_token = new_token.strip()
            new_token_id = new_token_id.view(1, 1)
            cur_txt = cur_txt + new_token

        return cur_txt
