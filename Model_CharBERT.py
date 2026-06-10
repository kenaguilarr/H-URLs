import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_utils import BertLayer, ACT2FN, BertPooler


BertLayerNorm = torch.nn.LayerNorm

class CharBERTModel(nn.Module):
    def __init__(self, config, is_roberta=False):
        super(CharBERTModel, self).__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self.char_embeddings = CharBertEmbeddings(config, is_roberta=is_roberta)

        self.encoder = CharBertEncoder(config, is_roberta=is_roberta)

        self.pooler = BertPooler(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, char_input_ids=None, start_ids=None, end_ids=None, input_ids=None, attention_mask=None,\
                token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,\
                encoder_attention_mask=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

                                                                                                         
                                                                                     
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
                                                                            
                                                                                              
                                                                                                                          
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    torch.long)                                                                       
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape,
                                                                                                        attention_mask.shape))

                                                                                 
                                                                                
                                                                        
                                                                              
                                                          
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)                      
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

                                                                          
                                                                                         
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape,
                        encoder_attention_mask.shape))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)                      
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

                                     
                                                    
                                                         
                                                                                  
                                                                                                               
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)                                           
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)                                                
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        char_embeddings = self.char_embeddings(char_input_ids, start_ids, end_ids)
        all_hidden_states_word, all_hidden_states_char = self.encoder(char_embeddings,
                                                                      embedding_output,
                                                                      attention_mask=extended_attention_mask,
                                                                      head_mask=head_mask,
                                                                      encoder_hidden_states=True,
                                                                      encoder_attention_mask=encoder_extended_attention_mask)

                                                                                                  
                                                      
        pooled_output = self.pooler(all_hidden_states_word[-1])

                                                                                                                      
                                                                  
                                                                       
        return all_hidden_states_word, all_hidden_states_char, pooled_output


class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)


        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings                     
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

bert_charemb_config = {"char_vocab_size": 1001,\
                       "char_embedding_size": 256,\
                       "kernel_size": 5}
class CharBertEmbeddings(nn.Module):
    def __init__(self, config, is_roberta=False):
        super(CharBertEmbeddings, self).__init__()
        self.config = config
        self.char_emb_config = bert_charemb_config
        self.char_embeddings = nn.Embedding(self.char_emb_config["char_vocab_size"],\
                                            self.char_emb_config["char_embedding_size"], padding_idx=0)
        self.rnn_layer = nn.GRU(input_size=self.char_emb_config["char_embedding_size"],\
                                hidden_size=int(config.hidden_size / 4), batch_first=True, bidirectional=True)

    def forward(self, char_input_ids, start_ids, end_ids):
        input_shape = char_input_ids.size()
                                                                                      
        assert len(input_shape) == 2

        batch_size, char_maxlen = input_shape[0], input_shape[1]

        char_input_ids_reshape = torch.reshape(char_input_ids, (batch_size, char_maxlen))
        char_embeddings = self.char_embeddings(char_input_ids_reshape)

        self.rnn_layer.flatten_parameters()
        all_hiddens, last_hidden = self.rnn_layer(char_embeddings)


        start_one_hot = nn.functional.one_hot(start_ids, num_classes=char_maxlen)

        end_one_hot = nn.functional.one_hot(end_ids, num_classes=char_maxlen)

        start_hidden = torch.matmul(start_one_hot.float(), all_hiddens)

        end_hidden = torch.matmul(end_one_hot.float(), all_hiddens)

        char_embeddings_repr = torch.cat([start_hidden, end_hidden], dim=-1)

        return char_embeddings_repr

class CharBertEncoder(nn.Module):
    def __init__(self, config, is_roberta=False):
        super(CharBertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = True
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.word_linear1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                                                                                          
        self.char_linear1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                                                                                          
                                                                         
        if not is_roberta:
            fusion_layer = torch.nn.Conv1d(in_channels=config.hidden_size * 2, out_channels=config.hidden_size,
                                           kernel_size=3, padding=3 // 2)
            self.fusion_layer_list = nn.ModuleList([fusion_layer for _ in range(config.num_hidden_layers)])
        else:
            self.fusion_layer = torch.nn.Conv1d(in_channels=config.hidden_size * 2, out_channels=config.hidden_size,
                                                kernel_size=3, padding=3 // 2)

        self.act_layer = ACT2FN[config.hidden_act]
        self.word_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.char_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.is_roberta = is_roberta

    def forward(self, char_hidden_states, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states_char = ()
        all_hidden_states_word = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            fusion_layer = None
            if not self.is_roberta:
                fusion_layer = self.fusion_layer_list[i]
            else:
                fusion_layer = self.fusion_layer

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask)
            char_layer_outputs = layer_module(char_hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                              encoder_attention_mask)

            word_outputs = layer_outputs[0]
            char_outputs = char_layer_outputs[0]
            word_transform = self.word_linear1(word_outputs)
            char_transform = self.char_linear1(char_outputs)
                                                                                               
            share_cat = torch.cat([word_transform, char_transform], dim=-1)
            share_permute = share_cat.permute(0, 2, 1)
            share_fusion = fusion_layer(share_permute)
            share_hidden = share_fusion.permute(0, 2, 1)

            hidden_states = self.word_norm(share_hidden + word_outputs)
            char_hidden_states = self.char_norm(share_hidden + char_outputs)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                                    
            if self.output_hidden_states:
                all_hidden_states_word = all_hidden_states_word + (hidden_states,)
                all_hidden_states_char = all_hidden_states_char + (char_hidden_states,)

                                                       
                                      
                                                                                      
                                   
                                                  
                                                                                                                         

        return  all_hidden_states_word, all_hidden_states_char