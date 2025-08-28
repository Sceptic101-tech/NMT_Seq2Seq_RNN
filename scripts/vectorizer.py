import numpy as np

class Seq2Seq_Vectorizer:
    '''Реализация алгоритмов векторизации'''
    def __init__(self, source_vocab, target_vocab, max_source_len : int, max_target_len : int):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def _vectorize(self, indices : list[int], forced_len : int=-1, mask_index : int=0) -> np.array:
        if forced_len <= 0:
            forced_len = len(indices)
        result_vec = np.empty(forced_len, dtype=np.int64)
        result_vec[:len(indices)] = indices
        result_vec[len(indices):] = mask_index

        return result_vec
    
    def _get_indices(self, tokens : list[str], add_bos : bool=False, add_eos : bool=False, is_target : bool=False) -> list[int]:
        indices = []
        cw_vocab = self.target_vocab if is_target else self.source_vocab

        if add_bos:
            indices.append(cw_vocab._bos_index)

        for token in tokens:
            indices.append(cw_vocab.get_token_index(token))
        
        if add_eos:
            indices.append(cw_vocab._eos_index)
        return indices
        

    def vectorize_vector_onehot(self, tokens : list[str], is_target=False) -> np.array:
        '''Принимает список токенов, возвращает onehot вектор, где "1" стоят в позиции индексов токенов'''
        cw_vocab = self.target_vocab if is_target else self.source_vocab
        onehot_vec = np.zeros(len(cw_vocab), dtype=np.float32)
        for token in tokens:
            onehot_vec[cw_vocab.get_token_index(token)] = 1
        return onehot_vec
    
    def vectorize(self, source_tokens : list[str], target_tokens : list[str]=None, use_dataset_max_len : bool=True):
        '''Векторизация текста для Seq2Seq модели'''
        max_source_len = self.max_source_len + 2 if use_dataset_max_len else -1
        max_target_len = self.max_target_len + 1 if use_dataset_max_len else -1

        source_indices = self._get_indices(source_tokens, add_bos=True, add_eos=True, is_target=False)
        source_vec = self._vectorize(source_indices, max_source_len)

        target_x_vec = target_y_vec = None

        if target_tokens is not None:
            target_x_indices = self._get_indices(target_tokens, add_bos=True, add_eos=False, is_target=True)
            target_x_vec = self._vectorize(target_x_indices, max_target_len)

            target_y_indices = self._get_indices(target_tokens, add_bos=False, add_eos=True, is_target=True)
            target_y_vec = self._vectorize(target_y_indices, max_target_len)


        return {'source_vec' : source_vec,
                'target_x_vec' : target_x_vec,
                'target_y_vec' : target_y_vec,
                'source_len' : len(source_indices)}


    @classmethod
    def from_dataframe(cls, texts_df, threshold_freq = 25):
        pass

    # @classmethod
    # def from_serializable(cls, serializable : dict):
    #     return Seq2Seq_Vectorizer(tokens_vocab=\
    #                       serializable['tokens_vocab'].from_serializable(),
    #                       label_vocab=\
    #                       serializable['label_vocab'].from_serializable())
    
    def to_serializable(self) -> dict:
        return {'tokens_vocab' : self.tokens_vocab.to_serializable(), 'label_vocab' : self.label_vocab.to_serializable()}