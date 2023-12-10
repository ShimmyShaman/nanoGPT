
ENDOFTEXT = "<|endoftext|>"
FILENAME = "<|filename|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

class OdinInfillEncoder():
    def __init__(self, name, vocabsize, pattern, mergeable_ranks, special_tokens) -> None:
        self.name = name
        self.vocabsize = vocabsize
        self.pattern = pattern
        self.mergeable_ranks = mergeable_ranks
        self.special_tokens = special_tokens

    def encode_single_token(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        return chr(2).encode('utf-8')
    
    def encode_ordinary(self, text):
        e = []
        for token in text:
            uc = ord(token)
            if uc > 127:
                uc = 2
            e.append(uc)
            # if token in self.special_tokens:
            #     e.append(self.special_tokens[token])
            # elif token in self.mergeable_ranks:
            #     e.append(self.mergeable_ranks[token])
            # else:
            #     e.append(chr(2).encode('utf-8'))
        return e