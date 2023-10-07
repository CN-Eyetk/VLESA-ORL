import torch
import torch.nn.functional as F
from torch import nn

class EmoTrans(nn.Module):
    def __init__(self, n_emo_in, n_emo_out, n_strat, embed_dim):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.n_emo_out = n_emo_out
        self.n_strat = n_strat
        self.embed_dim = embed_dim
        self.matrices = nn.ParameterList([nn.Parameter(torch.Tensor(n_emo_in, n_emo_out)) for i in range(n_strat)])
        self.emotion_embedding = nn.Embedding(n_emo_out, embed_dim)
        self.emotion_id = torch.tensor(range(n_emo_out), dtype=torch.long)
        self.dropout = nn.Dropout(0.1)
        self.reset_weights()
    def reset_weights(self):
        for weight in self.matrices:
            torch.nn.init.ones_(
                weight)
    def forward(self, emo_logits, strat_logits):
        b = emo_logits.size(0)
        emo_out_logits_each_strat = torch.zeros(b, self.n_strat, self.n_emo_out).to(emo_logits.device) #[b, stra, emo]
        emo_logits = self.dropout(emo_logits)
        strat_logits = self.dropout(strat_logits)
        emo_prob = F.softmax(emo_logits, dim = -1)
        for i,matrix in enumerate(self.matrices):
            with torch.no_grad():
                weight_norm = matrix/matrix.sum(dim=1, keepdim=True)
                matrix.copy_(weight_norm)
            emo_out_logits_cur_strat = F.linear(emo_prob, matrix.t())
            emo_out_logits_each_strat[:, i, :] = emo_out_logits_cur_strat
        strat_prob = F.softmax(strat_logits, dim = -1)
        emo_out_prob = torch.bmm(strat_prob.unsqueeze(-2), emo_out_logits_each_strat) #[b, 1, stra] * [b, stra, emo] -> [b, 1, emo] 
        emotion_id = self.emotion_id.to(emo_logits.device) 
        emo_embed = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1))
        emo_out_prob = emo_out_prob.squeeze()
        emo_out_prob = torch.log(emo_out_prob) #upDATE  9-27-II
        return emo_embed, emo_out_prob
        

class EmoTrans_Pro(nn.Module):
    def __init__(self, n_emo_in, n_emo_out, n_strat, embed_dim):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.n_emo_out = n_emo_out
        self.n_strat = n_strat
        self.embed_dim = embed_dim
        self.matrices = nn.ParameterList([nn.Parameter(torch.Tensor(n_emo_in, n_emo_out)) for i in range(n_strat)])
        #self.emotion_embedding = nn.Embedding(n_emo_out, embed_dim)
        self.emotion_id = torch.tensor(range(n_emo_out), dtype=torch.long)
        self.reset_weights()
    def reset_weights(self):
        for weight in self.matrices:
            torch.nn.init.ones_(
                weight)
    def forward(self, emo_logits, strat_logits, emotion_embedding):
        b = emo_logits.size(0)
        emo_out_logits_each_strat = torch.zeros(b, self.n_strat, self.n_emo_out).to(emo_logits.device) #[b, stra, emo]
        emo_prob = F.softmax(emo_logits, dim = -1)
        for i,matrix in enumerate(self.matrices):
            with torch.no_grad():
                weight_norm = matrix/matrix.sum(dim=1, keepdim=True)
                matrix.copy_(weight_norm)
            emo_out_logits_cur_strat = F.linear(emo_prob, matrix.t())
            emo_out_logits_each_strat[:, i, :] = emo_out_logits_cur_strat
        strat_prob = F.softmax(strat_logits, dim = -1)
        #print(strat_prob)
        emo_out_prob = torch.bmm(strat_prob.unsqueeze(-2), emo_out_logits_each_strat) #[b, 1, stra] * [b, stra, emo] -> [b, 1, emo] 
        #emo_out_prob = F.softmax(emo_out_logits, dim = -1) #[b, 1, emo]
        #print(emo_out_prob)
        #emotion_id = self.emotion_id.to(emo_logits.device) 
        emo_embed = torch.bmm(emo_out_prob,  emotion_embedding.unsqueeze(0).repeat(b, 1, 1)) #use transformer decoder weight
        emo_out_prob = emo_out_prob.squeeze()
        return emo_embed, emo_out_prob

class CatAttention(nn.Module):
    
    def __init__(self, n_hidden_in, n_hidden_out):
        
        super().__init__()
        
        self.h_hidden_enc = n_hidden_in
        self.h_hidden_dec = n_hidden_out
        
        self.W = nn.Linear(2*n_hidden_in, n_hidden_out, bias=False) 
        self.V = nn.Parameter(torch.rand(n_hidden_out))
        
    
    def forward(self, hidden_targ, hidden_src, mask):
        ''' 
            PARAMS:           
                hidden_dec:     [b, n_hidden_dec]    (1st hidden_dec = encoder's last_h's last layer)                 
                last_layer_enc: [b, seq_len, n_hidden_enc * 2] 
            
            RETURN:
                att_weights:    [b, src_seq_len] 
        '''

        batch_size = hidden_src.size(0)
        src_seq_len = hidden_src.size(1)

        hidden_targ_new = hidden_targ.unsqueeze(1).repeat(1, src_seq_len, 1)         #[b, src_seq_len, n_hidden_dec]
        #print("hidden_targ_new", hidden_targ_new.shape)
        #print("hidden_src", hidden_src.shape)
        tanh_W_s_h = torch.tanh(self.W(torch.cat((hidden_targ_new, hidden_src), dim=2)))  #[b, src_seq_len, n_hidden_dec]
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)       #[b, n_hidde_dec, seq_len]
        
        V = self.V.repeat(batch_size, 1).unsqueeze(1)  #[b, 1, n_hidden_dec]
        e = torch.bmm(V, tanh_W_s_h).squeeze(1)        #[b, seq_len]
        #mask[mask == 0] = -1e8
        #e = e*mask
        att_weights = F.softmax(e, dim=1)              #[b, src_seq_len]
        att_weights = att_weights * mask
        hidden_output = torch.bmm(att_weights.unsqueeze(1), hidden_src).squeeze(1)        
        hidden_output = hidden_output + hidden_targ
        return att_weights, hidden_output

if __name__ == "__main__":
    n_emo_in = 3
    n_emo_out = 4
    n_strat = 5
    batch_size = 2
    tran = EmoTrans(n_emo_in, n_emo_out, n_strat, embed_dim = 32)
    emo_logits = torch.full((batch_size, n_emo_in), 3.1)
    strat_logits = torch.full((batch_size, n_strat), 2.2)
    emo_embed, emo_out_logits = tran(emo_logits, strat_logits)