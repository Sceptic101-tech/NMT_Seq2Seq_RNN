import numpy as np


class Seq2Seq_Vectorizer:
    """
    Класс, отвечающий за преобразование списка токенов в числовые массивы,
    которые можно подавать на вход модели seq2seq.
    """

    def __init__(self, source_vocab, target_vocab, max_source_len: int, max_target_len: int):
        """
        Инициализация:
            * `source_vocab` – Vocabulary для исходного языка
            * `target_vocab` – Vocabulary для целевого языка
            * `max_source_len` – максимальная длина источника (без BOS/EOS)
            * `max_target_len` – максимальная длина цели (без BOS/EOS)
        """
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def _vectorize(self, indices: list[int], forced_len: int = -1, mask_index: int = 0) -> np.array:
        """
        Заполняет массив индексов до `forced_len` нулями (или другим индексом).
        Если `forced_len <= 0`, использует фактическую длину списка.
        """
        if forced_len <= 0:
            forced_len = len(indices)
        result_vec = np.empty(forced_len, dtype=np.int64)
        result_vec[:len(indices)] = indices
        result_vec[len(indices):] = mask_index

        return result_vec
    
    def _get_indices(self, tokens: list[str], add_bos: bool = False,
                     add_eos: bool = False, is_target: bool = False) -> list[int]:
        """
        Преобразует список токенов в индексы.
        При `add_bos=True` добавляется индекс BOS в начало,
        при `add_eos=True` – индекс EOS в конец.
        Для целевого языка берётся target_vocab, иначе source_vocab.
        """
        indices = []
        cw_vocab = self.target_vocab if is_target else self.source_vocab

        if add_bos:
            indices.append(cw_vocab._bos_index)

        for token in tokens:
            indices.append(cw_vocab.get_token_index(token))
        
        if add_eos:
            indices.append(cw_vocab._eos_index)
        return indices
        

    def vectorize_vector_onehot(self, tokens: list[str], is_target=False) -> np.array:
        """
        Преобразует список токенов в one‑hot вектор.
        В позиции индекса токена ставится 1 (все остальные – 0).
        """
        cw_vocab = self.target_vocab if is_target else self.source_vocab
        onehot_vec = np.zeros(len(cw_vocab), dtype=np.float32)
        for token in tokens:
            onehot_vec[cw_vocab.get_token_index(token)] = 1
        return onehot_vec
    
    def vectorize(self, source_tokens: list[str], target_tokens: list[str] = None,
                  use_dataset_max_len: bool = True):
        """
        * `source_tokens` – токены исходного текста.
        * `target_tokens` – токены целевого текста (может быть None, если нужен только encoder).
        * `use_dataset_max_len` – если True, добавляем к длине +2 для BOS/EOS
          и +1 для target (плюс EOS), иначе используем реальную длину.

        Возвращает словарь:
            source_vec      – вектор исходного текста (BOS+tokens+EOS)
            target_x_vec    – вход декодера (BOS+target без EOS)
            target_y_vec    – выход декодера (target без BOS + EOS)
            source_len      – фактическая длина source_tokens (+2, если включено BOS/EOS)
        """
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
    def from_dataframe(cls, texts_df, threshold_freq=25):
        """
        Строит объект Seq2Seq_Vectorizer из DataFrame.
        В DataFrame ожидаются колонки:
            * source_text – исходный текст
            * target_text – целевой текст
            * split       – раздел (train/validation/test)

        Функция должна собрать словари для source и target,
        определить max_source_len / max_target_len и вернуть готовый векторизатор.
        """
        # TODO: Реализовать построение из DataFrame
        pass
    
    def to_serializable(self) -> dict:
        return {'tokens_vocab' : self.tokens_vocab.to_serializable(), 'label_vocab' : self.label_vocab.to_serializable()}