
import Lark
import regex


class CFG:

    def __init__(self, tokenizer, template) -> None:

        self.vocab = tokenizer.get_vocab()
        self.all_tokens = self.vocab.keys()
        self.parser = Lark(template, parser='lalr',
                           lexer='basic',
                           propagate_positions=False,
                           maybe_placeholders=False,
                           regex=True)
        self.regex_map = self.__build_pattern_dict__()

    def __build_pattern_dict__(self):
        regex_map = {}
        for term in self.parser.terminals:
            if term.pattern:
                regex_map[term.name] = regex.compile(term.pattern.to_regexp())
        return regex_map

    def __full_match__(self, patterns):
        accept_tokens = []
        for p in patterns:
            for t in self.all_tokens:
                if self.regex_map[p].fullmatch(t):
                    accept_tokens.append(t)
        accept_token_ids = [self.vocab[t] for t in accept_tokens]
        return accept_token_ids, accept_tokens

    def __context_match__(self, cur_txt, patterns):
        accept_tokens = []
        for p in patterns:
            for t in self.all_tokens:
                candidate = cur_txt+t
                m = self.regex_map[p].search(candidate)
                if m and candidate.endswith(m.group()):
                    accept_tokens.append(t)
        accept_token_ids = [self.vocab[t] for t in accept_tokens]
        return accept_token_ids, accept_tokens

    def get_accept_token_ids(self, cur_txt):
        interactive = self.json_parser.parse_interactive(cur_txt)
        interactive.exhaust_lexer()
        results = interactive.accepts()
        if '$END' in results:
            return None, None
        accept_token_ids, accept_tokens = self.__full_match__(results)
        if len(accept_token_ids) == 0:
            accept_token_ids, accept_tokens = self.__context_match__(
                cur_txt, results)

        return accept_token_ids, accept_tokens
