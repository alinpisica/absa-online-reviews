import torch
import torch.nn as nn

class BERT_BiLSTM(nn.Module):
    def __init__(self, bert_model, device):
        super().__init__()          
        
        self.device = device

        self.embedding = bert_model

        self.dropout = torch.nn.Dropout(0.3)

        self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(256, 3)
    
    def forward(self, ids, mask, token_type_ids):
        _, embedded = self.embedding(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)

        output_2 = self.dropout(embedded.to(self.device)).to(self.device)

        output_3 = self.lstm(output_2.unsqueeze(1))
        
        fc_out = self.fc(output_3[0]).squeeze(1)

        return fc_out