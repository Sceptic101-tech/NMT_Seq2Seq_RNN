import regex as re

class SeparatorTokenizer:
    '''Простейшая реализация алгоритма токенизации. Разделение происходит по separator'''
    def __init__(self):
        pass

    def tokenize(self, text : str, separator : str = None) -> list[str]:
        text = re.sub(r'([^\w\s]|_)', r' \1 ', text) # Отделяем пробелом знаки препинаня, они будут считаться отдельным токеном
        text = re.sub(r'[\t\n\r\f\v]', r' ', text)
        return text.split(separator)