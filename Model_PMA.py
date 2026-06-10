import os
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
from attention import CBAMLayer
from transformers import BertConfig
import torch.nn.functional as F
from Model_CharBERT import CharBERTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHARBERT_MODEL_DIR = os.path.join("charbert-bert-wiki", "charbert-bert-wiki")


class CharBertModel(nn.Module):

    def __init__(self):
        super(CharBertModel, self).__init__()
        config = BertConfig.from_pretrained(CHARBERT_MODEL_DIR)
        self.bert = CharBERTModel(config)
        checkpoint = torch.load(
            os.path.join(CHARBERT_MODEL_DIR, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        self.load_state_dict(checkpoint, strict=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)                       
        self.fc = nn.Linear(768, 2)
        self.hidden_size = 768
        self.fuse = nn.Conv1d(2 * self.hidden_size, self.hidden_size, kernel_size=1)
        self.cbam = CBAMLayer(channel=12)

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]
        device = context.device

                                    
        char_ids = x[3]
        start_ids = x[4]
        end_ids = x[5]

                                                 
                   
                                                                                                               
                                                                                             
        all_hidden_states_word, all_hidden_states_char, pooled_output = self.bert(
            char_input_ids=char_ids,
            start_ids=start_ids,
            end_ids=end_ids,
            input_ids=context,
            attention_mask=mask,
            token_type_ids=types,
        )

                                             
        fuse_output = []

        for x1, x2 in zip(all_hidden_states_word, all_hidden_states_char):
            x1 = x1.to(device)
            x2 = x2.to(device)

                                     
            x = torch.cat([x1, x2], dim=-1)                                  

                                
            x = x.view(x.size(0), -1, x.size(2))                                  

                                               
            y = self.fuse(x.transpose(1, 2))                                                                                                            

                                
            y_output = y.transpose(1, 2)                                       

                                                  
            fuse_output.append(y_output)

        pyramid = tuple(fuse_output)
        pyramid = torch.stack(pyramid, dim=0).permute(1, 0, 2, 3)
                                        

        pos_pooled = self.cbam(pyramid)
                                        

        pyramid_levels = [1, 2, 3, 4]                               
        output_feature_size = 768                       

                                                            
        pooled_features = []

        for level in pyramid_levels:
                                                              
            window_size = pos_pooled.size(1) // level

                                                
                                                                                                                        
            pooled_feature_tensor = F.avg_pool2d(pos_pooled.permute(0, 3, 2, 1), (1, window_size)).permute(0, 3, 2, 1)
                                            

                                                                
            pooled_features.append(pooled_feature_tensor)

                                                                             
        concatenated_features = torch.cat(pooled_features, dim=1)

                               
        compressed_feature_tensor = torch.mean(concatenated_features, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)

        out = self.dropout(compressed_feature_tensor)
        out = self.fc(out)

        return pyramid, pooled_output, out


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(CHARBERT_MODEL_DIR)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)                       
        self.fc = nn.Linear(768, 2)
        self.cbam = CBAMLayer(channel=12)

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]
        device = context.device

                                                        
                                        
        outputs, pooled = self.bert(input_ids=context, token_type_ids=types,
                                    attention_mask=mask,
                                    output_all_encoded_layers=True)

        pyramid = tuple(outputs)
        pyramid = torch.stack(pyramid, dim=0).permute(1, 0, 2, 3)
                                        

        pos_pooled = self.cbam(pyramid)
                                        

        pyramid_levels = [1, 2, 3, 4]                               
        output_feature_size = 768                       

                                                            
        pooled_features = []

        for level in pyramid_levels:
                                                              
            window_size = pos_pooled.size(1) // level

                                                
                                                                                                                        
            pooled_feature_tensor = F.avg_pool2d(pos_pooled.permute(0, 3, 2, 1), (1, window_size)).permute(0, 3, 2, 1)
                                            

                                                                
            pooled_features.append(pooled_feature_tensor)

                                                                                                   
        concatenated_features = torch.cat(pooled_features, dim=1)

                               
        compressed_feature_tensor = torch.mean(concatenated_features, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)

        out = self.dropout(compressed_feature_tensor)
        out = self.fc(out)                                      

        return pyramid, pooled, out
