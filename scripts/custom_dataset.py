class CustomDataset:
    def __init__(self, dataframe, tokenizer, vectorizer):
        self._vectorizer = vectorizer
        self._tokenizer = tokenizer

        self._main_df = dataframe

        self._train_df = self._main_df[self._main_df.split == 'train']
        self._train_len = len(self._train_df)

        self._valid_df = self._main_df[self._main_df.split == 'validation']
        self._valid_len = len(self._valid_df)

        self._test_df = self._main_df[self._main_df.split == 'test']
        self._test_len = len(self._test_df)

        self._lookup_split = {'train' : (self._train_df, self._train_len),\
                              'validation' : (self._valid_df, self._valid_len),\
                              'test' : (self._test_df, self._test_len)}
        
        self.set_dataframe_split('train')

    def __getitem__(self, index):
        row = self._cw_dataframe.iloc[index]
        vector_dict = self._vectorizer.vectorize(source_tokens=row['source_text'], target_tokens=row['target_text'], use_dataset_max_len=True)
        return {
            'source_vec': vector_dict['source_vec'],
            'target_x_vec': vector_dict['target_x_vec'],
            'target_y_vec': vector_dict['target_y_vec'],
            'source_len': vector_dict['source_len']
        }
    
    def __len__(self):
        return self._cw_df_len
    
    def set_dataframe_split(self, split='train'):
        '''Set a current data split. Allowed values: train, test, validation'''
        self._cw_dataframe, self._cw_df_len = self._lookup_split[split]