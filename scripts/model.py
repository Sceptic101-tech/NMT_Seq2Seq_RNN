import numpy as np
import gc
import pandas
import time
import pandas as pd
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def dot_product_attention(encoder_state_vectors : torch.tensor, query_vector : torch.tensor):
    """
    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
        query_vector (torch.Tensor): hidden state
    """
    # query_vector.size() = [B, D]
    # encoder_state_vectors.size() = [B, N, D]
    # print(f'encoder_state_vectors.size() {encoder_state_vectors.size()}')
    # print(f'query_vector.size() {query_vector.size()}')
    vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)) # [B, N, 1]
    # print(f'before squeeze vector_scores.size() {vector_scores.size()}')
    vector_scores = vector_scores.squeeze(dim=-1) # [B, N]
    # print(f'after squeeze vector_scores.size() {vector_scores.size()}')
    vector_probabilities = F.softmax(vector_scores, dim=-1) # [B, N]
    # print(f'vector_probabilities.size() {vector_probabilities.size()}')
    encoder_state_vectors = encoder_state_vectors.transpose(-2, -1) # [B, D, N]
    context_vectors = torch.matmul(encoder_state_vectors, vector_probabilities.unsqueeze(dim=2)).squeeze(dim=-1) # [B, D]
    # print(f'context_vectors.size() {context_vectors.size()}')
    return context_vectors, vector_probabilities

class RNNEncoder(nn.Module):
    def __init__(self, num_embeddings : int, embedding_size : int, rnn_hidden_size : int):
        super().__init__()
    
        self.source_embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.birnn = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True, batch_first=True)
    
    def forward(self, x_source : torch.tensor, x_lengths : torch.tensor):
        """
        Args:
            x_source (torch.Tensor): the input data tensor.
                x_source.shape is (batch, seq_size)
            x_lengths (torch.Tensor): a vector of lengths for each item in the batch
        Returns:
            a tuple: x_unpacked (torch.Tensor), x_birnn_h (torch.Tensor)
                x_unpacked.shape = (batch, seq_size, rnn_hidden_size * 2)
                x_birnn_h.shape = (batch, rnn_hidden_size * 2)
        """
        # print(f'x_source.size() {x_source.size()}')
        x_embedded = self.source_embedding(x_source)
        # print(f'x_embedded.size() {x_embedded.size()}')
        # create PackedSequence; x_packed.data.shape=(number_items, embeddign_size)
        x_packed = pack_padded_sequence(x_embedded, x_lengths.detach().cpu().numpy(), 
                                        batch_first=True)
        
        # x_birnn_h.shape = (num_rnn, batch_size, feature_size)
        x_birnn_out, x_birnn_h  = self.birnn(x_packed)
        # permute to (batch_size, num_rnn, feature_size)
        x_birnn_h = x_birnn_h.permute(1, 0, 2)
        
        # flatten features; reshape to (batch_size, num_rnn * feature_size)
        x_birnn_h = x_birnn_h.reshape(x_birnn_h.size(0), -1) # Конкатенация последних скрытых состояний из двух RNN
        
        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True) # Выходы сети в каждый момент времени. Поскольку сеть двунаправленная, 
        # print(f'x_unpacked.size() {x_unpacked.size()}')
        # print(f'x_birnn_h.size() {x_birnn_h.size()}')
        
        return x_unpacked, x_birnn_h


class RNNDecoder(nn.Module):
    def __init__(self, num_embeddings : int, embedding_size : int, rnn_hidden_size : int, fc_hidden_size : int, bos_index : int):
        """
        Args:
            num_embeddings (int): number of embeddings is also the number of 
                unique words in target vocabulary 
            embedding_size (int): the embedding vector size
            rnn_hidden_size (int): size of the hidden rnn state
            bos_index(int): begin-of-sequence index
        """
        super().__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                             embedding_dim=embedding_size,
                                             padding_idx=0)
        self.gru_cell = nn.GRUCell(embedding_size + rnn_hidden_size,
                                   rnn_hidden_size)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        # self.classifier = nn.Linear(rnn_hidden_size*2, num_embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size*2, fc_hidden_size),
            nn.ELU(),
            nn.Linear(fc_hidden_size, num_embeddings)
        )
        self.bos_index = bos_index
    
    def _init_indices(self, batch_size : int):
        '''Возвращает ввектор, заполненный индексом токена BOS'''
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index
    
    def _init_context_vectors(self, batch_size : int):
        """Возвращает нулевой вектор"""
        return torch.zeros(batch_size, self._rnn_hidden_size)
            
    def forward(self, encoder_state, initial_hidden_state, target_sequence=None, forced_batch_size=None, sample_probability=0.0, output_sequence_size=0, temperature=1):
        """  
        Args:
            encoder_state (torch.Tensor): the output of the NMTEncoder
            initial_hidden_state (torch.Tensor): The last hidden state in the  NMTEncoder
            target_sequence (torch.Tensor): the target text data tensor
            sample_probability (float): the schedule sampling parameter
                probabilty of using model's predictions at each decoder step.
                1: using only model prediction
                0: using only target sequence
        Returns:
            output_vectors (torch.Tensor): prediction vectors at each output step
        """
        if target_sequence is None:
            sample_probability = 1.0
        else:
            # We are making an assumption there: The batch is on first
            # The input is (Batch, Seq)
            # We want to iterate over sequence so we permute it to (S, B)
            target_sequence = target_sequence.permute(1, 0)
            output_sequence_size = target_sequence.size(0)
        
        # use the provided encoder hidden state as the initial hidden state
        h_t = self.hidden_map(initial_hidden_state)
        
        # Насильно меняем размер батча (используется прри генерации)
        if forced_batch_size is None:
            batch_size = encoder_state.size(0)
        else:
            batch_size = forced_batch_size
        # initialize context vectors to zeros
        context_vectors = self._init_context_vectors(batch_size)
        # initialize first y_t word as BOS
        y_t_index = self._init_indices(batch_size)
        
        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()
        
        for i in range(output_sequence_size):
            # Позволяем модели использовать собственные предсказания в качестве ground-truth. Это помогает модели "искать" истинную зависимость
            use_sample = np.random.random() < sample_probability
            if not use_sample:
                y_t_index = target_sequence[i]
                
            # Слой эмбеддинга и конкатенация с прошлым вектором контекста
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)
            
            # Получаем новое скрытое состояние
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().detach().numpy()) # Для откладки
            
            # Используя текущее скрытое состояние как вектор запроса, обноляем конекст
            context_vectors, p_attn = dot_product_attention(encoder_state_vectors=encoder_state,\
                                                               query_vector=h_t)
            
            # Кэширование вероятностей внимания
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())
            
            # Предсказываем следующий токен на основе текущего скрытого состояния и контекста
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(F.dropout(prediction_vector, 0.3))
            
            if use_sample:
                p_y_t_index = F.softmax(score_for_y_t_index * temperature, dim=1)
                # _, y_t_index = torch.max(p_y_t_index, 1)
                y_t_index = torch.multinomial(p_y_t_index, 1).squeeze(dim=1)
            
            # Кэширование вероятностей токенов
            output_vectors.append(score_for_y_t_index)
            
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        
        return output_vectors


class Seq2Seq_Model(nn.Module):
    def __init__(self, source_vocab_size : int, source_embedding_size : int, 
                 target_vocab_size : int, target_embedding_size : int, encoder_rnn_size : int, 
                 fc_hidden_size : int, target_bos_index : int):
        """
        Args:
            source_vocab_size (int): number of unique words in source language
            source_embedding_size (int): size of the source embedding vectors
            target_vocab_size (int): number of unique words in target language
            target_embedding_size (int): size of the target embedding vectors
            encoding_size (int): the size of the encoder RNN.  
        """
        super().__init__()
        self.encoder = RNNEncoder(num_embeddings=source_vocab_size, 
                                  embedding_size=source_embedding_size,
                                  rnn_hidden_size=encoder_rnn_size)
        
        decoding_size = encoder_rnn_size * 2
        
        self.decoder = RNNDecoder(num_embeddings=target_vocab_size, 
                                  embedding_size=target_embedding_size, 
                                  rnn_hidden_size=decoding_size,
                                  fc_hidden_size=fc_hidden_size,
                                  bos_index=target_bos_index)
    
    def forward(self, x_source, x_source_lengths, target_sequence, sample_probability=0.0):
        """The forward pass of the model
        
        Args:
            x_source (torch.Tensor): the source text data tensor. 
                x_source.shape should be (batch, vectorizer.max_source_length)
            x_source_lengths torch.Tensor): the length of the sequences in x_source 
            target_sequence (torch.Tensor): the target text data tensor
            sample_probability (float): the schedule sampling parameter
                probabilty of using model's predictions at each decoder step
        Returns:
            decoded_states (torch.Tensor): prediction vectors at each output step
        """
        encoder_state, final_hidden_states = self.encoder(x_source, x_source_lengths)
        decoded_states = self.decoder(encoder_state=encoder_state, 
                                      initial_hidden_state=final_hidden_states, 
                                      target_sequence=target_sequence, 
                                      sample_probability=sample_probability)
        return decoded_states
