import pandas
import json

class Vocabulary:
    '''Словарь токенов и их индексов'''
    def __init__(self, token_to_idx : dict = None, mask_token : str = '<MASK>', unk_token : str = '<UNK>',\
                 bos_token : str = '<BOS>', eos_token : str = '<EOS>', is_lexical_tokens : bool=True):
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx
        self._idx_to_token = {value : key for key, value in token_to_idx.items()}

        self._is_lexical_tokens = is_lexical_tokens
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        if is_lexical_tokens:
            self.mask_token_index = self.add_token(mask_token)
            self.unk_index = self.add_token(self.unk_token)
            self._bos_index = self.add_token(self.bos_token)
            self._eos_index = self.add_token(self.eos_token)

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def to_serializable(self) -> dict:
        return {'token_to_idx' : self.token_to_idx, 'mask_token' : self.mask_token,\
                'unk_token' : self.unk_token, 'bos_token' : self.bos_token, 'eos_token' : self.eos_token, 'is_lexical_tokens' : self._is_lexical_tokens}
    
    def to_json(self, filepath : str):
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(self.to_serializable(), file, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath : str):
        with open(filepath, encoding='utf-8') as file:
            return cls.from_serializable(json.load(file))

    @classmethod
    def from_serializable(cls, serializable : dict):
        return cls(**serializable)

    @classmethod
    def from_dataframe(cls, dataframe : pandas.DataFrame, tokenizer, treshold_freq=25):
        pass
    
    def add_token(self, token : str) -> int:
        '''Добавить токен в словарь'''
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self._idx_to_token[idx] = token
            return idx
        else:
            return self.token_to_idx[token]

    def add_tokens(self, tokens : list[str]) -> list[int]:
        '''Добавить список токенов в словарь'''
        return [self.add_token(token) for token in tokens]

    def get_token_index(self, token : str) -> int:
        '''Возвращает индекс токена или unk_index'''
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        else:
            return self.unk_index

    def get_token(self, index : int) -> str:
        '''Возвращает токен, соответствующий индексу, или unk_token'''
        if index in self._idx_to_token:
            return self._idx_to_token[index]
        else:
            return self.unk_token
        
    def size(self):
        return len(self.token_to_idx)