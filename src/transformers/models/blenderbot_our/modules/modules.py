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

class EmoTransVAE(nn.Module):
    def __init__(self, config, n_emo_in, n_emo_out, n_strat, embed_dim):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.n_emo_out = n_emo_out
        self.n_strat = n_strat
        self.embed_dim = embed_dim
        #self.matrices = nn.ParameterList([nn.Parameter(torch.Tensor(n_emo_in, n_emo_out)) for i in range(n_strat)])
        self.emotion_embedding = nn.Embedding(n_emo_out, embed_dim)
        self.emotion_id = torch.tensor(range(n_emo_out), dtype=torch.long)
        self.dropout = nn.Dropout(0.1)
        
        self.hidden_dim = config.d_model
        self.latent_dim = int(self.hidden_dim /2)
        
        self.h_prior_emo = nn.Linear(self.hidden_dim + self.n_emo_in, self.latent_dim)
        self.h_prior_strat = nn.Linear(self.hidden_dim + self.n_strat, self.latent_dim)
        self.mu_prior = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar_prior = nn.Linear(self.hidden_dim, self.latent_dim)
        self.Dense_z_prior = nn.Linear(self.latent_dim, self.n_emo_out)

        self.h_posterior_emo = nn.Linear(self.hidden_dim + self.n_emo_in + self.n_emo_out, self.latent_dim)
        self.h_posterior_strat = nn.Linear(self.hidden_dim + self.n_strat + self.n_emo_out, self.latent_dim)
        self.mu_posterior = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar_posterior = nn.Linear(self.hidden_dim, self.latent_dim)
        self.Dense_z_posterior = nn.Linear(self.latent_dim, self.n_emo_out)
        

    def prior(self, hidden_emo, hidden_strat, p_emo_in, p_strat):
        h1_emo = F.relu(self.h_prior_emo(torch.cat((hidden_emo, p_emo_in), dim = -1)))
        h1_strat = F.relu(self.h_prior_strat(torch.cat((hidden_strat, p_strat), dim = -1)))
        mu = self.mu_prior(h1_emo)
        logvar = self.logvar_prior(h1_strat)
        return mu, logvar
    
    def posterior(self, hidden_emo, hidden_strat, p_emo_in, p_strat, p_emo_out):
        h1_emo = F.relu(self.h_posterior_emo(torch.cat((hidden_emo, p_emo_in, p_emo_out), dim = -1)))
        h1_strat = F.relu(self.h_posterior_strat(torch.cat((hidden_strat, p_strat, p_emo_out), dim = -1)))
        mu = self.mu_posterior(h1_emo)
        logvar = self.logvar_posterior(h1_strat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def foward(self, hidden_emo, hidden_strat, p_emo_in, p_strat):
        b = emo_logits.size(0)
        mu, logvar = self.prior(hidden_emo, hidden_strat, p_emo_in, p_strat)
        z = self.reparameterize(mu, logvar)
        E_prob = torch.softmax(self.Dense_z_prior_positive(z), dim=-1).unsqueeze(-2) #[b, 1,n_emo]
        emotion_id = self.emotion_id.to(hidden_emo.device) 
        emo_out_emb = torch.bmm(E_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1))
        E_prob = torch.log(E_prob.squeeze(1))
        return emo_out_emb, mu, logvar, E_prob
    
    def forward_train(self,  hidden_emo, hidden_strat, p_emo_in, p_strat, p_emo_out):
        b = emo_logits.size(0)
        mu, logvar = self.posterior(hidden_emo, hidden_strat, p_emo_in, p_strat, p_emo_out)
        z = self.reparameterize(mu, logvar)
        E_prob = torch.softmax(self.Dense_z_prior_positive(z), dim=-1).unsqueeze(-2) #[b, 1,n_emo]
        emotion_id = self.emotion_id.to(hidden_emo.device) 
        emo_out_emb = torch.bmm(E_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1)) 
        E_prob = torch.log(E_prob.squeeze(1))
        return emo_out_emb, mu, logvar, E_prob
    @staticmethod
    def kl_div(mu_posterior, logvar_posterior, mu_prior=None, logvar_prior=None):
        """
        This code is adapted from:
        https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/83ca9dd96272d3a38978a1dfa316d06d5d6a7c77/model/utils/probability.py#L20
        """
        one = torch.FloatTensor([1.0]).to(mu_posterior.device)
        if mu_prior == None:
            mu_prior = torch.FloatTensor([0.0]).to(mu_posterior.device)
            logvar_prior = torch.FloatTensor([0.0]).to(logvar_posterior.device)
        kl_div = torch.sum(0.5 * (logvar_prior - logvar_posterior + (logvar_posterior.exp()+(mu_posterior-mu_prior).pow(2))/logvar_prior.exp() - one) )
        return kl_div
        


class EmoTransVAE_MultiStrat(nn.Module):
    def __init__(self, config, n_emo_in, n_emo_out, n_strat, embed_dim):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.n_emo_out = n_emo_out
        self.n_strat = n_strat
        self.embed_dim = embed_dim
        #self.matrices = nn.ParameterList([nn.Parameter(torch.Tensor(n_emo_in, n_emo_out)) for i in range(n_strat)])
        self.emotion_embedding = nn.Embedding(n_emo_out, embed_dim)
        self.emotion_id = torch.tensor(range(n_emo_out), dtype=torch.long)
        self.dropout = nn.Dropout(0.1)
        
        self.hidden_dim = config.d_model
        self.latent_dim = int(self.hidden_dim /2)
        
        self.h_prior_emo = nn.Linear(self.hidden_dim + self.n_emo_in, self.latent_dim)
        self.mu_priors = nn.ModuleList([nn.Linear(self.hidden_dim, self.latent_dim)  for i in range(self.n_strat)])
        self.logvar_priors = nn.ModuleList([nn.Linear(self.hidden_dim, self.latent_dim)  for i in range(self.n_strat)])
        self.Dense_z_prior = nn.Linear(self.latent_dim, self.n_emo_out)

        self.h_posterior_emo = nn.Linear(self.hidden_dim + self.n_emo_in + self.n_emo_out, self.latent_dim)
        #self.h_posterior_strat = nn.Linear(self.hidden_dim + self.n_strat + self.n_emo_out, self.latent_dim)
        self.mu_posteriors = nn.ModuleList([nn.Linear(self.hidden_dim, self.latent_dim)  for i in range(self.n_strat)])
        self.logvar_posteriors = nn.ModuleList([nn.Linear(self.hidden_dim, self.latent_dim)  for i in range(self.n_strat)])
        self.Dense_z_posterior = nn.Linear(self.latent_dim, self.n_emo_out) 
        

    def prior(self, hidden_emo, p_emo_in):
        b = hidden_emo.size(0)
        h1_emo = F.relu(self.h_prior_emo(torch.cat((hidden_emo, p_emo_in), dim = -1)))
        mus = torch.zeros((b, self.latent_dim, self.n_strat))
        logvars = torch.zeros((b, self.latent_dim, self.n_strat))
        for i in range(self.mu_priors):
            mu = self.mu_priors[i](h1_emo)
            logvar = self.logvar_priors[i](h1_emo)
            mus[:,:,i] = mu
            logvars[:, :, i] = logvar
        return mus, logvars
    
    def posterior(self, hidden_emo, p_emo_in, p_emo_out):
        b = hidden_emo.size(0)
        h1_emo = F.relu(self.h_posterior_emo(torch.cat((hidden_emo, p_emo_in, p_emo_out), dim = -1)))
        #h1_strat = F.relu(self.h_posterior_strat(torch.cat((hidden_strat, p_strat, p_emo_out), dim = -1)))
        mus = torch.zeros((b, self.latent_dim, self.n_strat)).to(h1_emo.device)
        logvars = torch.zeros((b, self.latent_dim, self.n_strat)).to(h1_emo.device)
        for i in range(self.mu_priors):
            mu = self.mu_priors[i](h1_emo)
            logvar = self.logvar_priors[i](h1_emo)
            mus[:,:,i] = mu
            logvars[:, :, i] = logvar
        return mus, logvars

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def foward(self, hidden_emo, p_emo_in, p_strat):
        b = emo_logits.size(0)
        mus, logvars = self.prior(hidden_emo, p_emo_in)
        zs = torch.zeros((b, self.latent_dim, self.n_strat)).to(hidden_emo.device)
        E_probs = torch.zeros((b, self.n_emo_out, self.n_strat)).to(hidden_emo.device) #[b, n_e_out, n_s]
        for i, (mu, logvar) in enumerate(zip(mus, logvars)):
            z = self.reparameterize(mu, logvar)
            zs[:,:,i] = z
            E_prob = torch.softmax(self.Dense_z_prior_positive(z), dim=-1)
            E_probs[:,:,i] = E_prob
        emo_out_prob = torch.bmm(E_probs, p_strat.unsqueeze(-1)).permute(0,2,1) #[b, n_e_out, n_s] [b,  n_s,1]  -> [b, n_e_out, 1] ->[b, 1, n_e_out]
        emotion_id = self.emotion_id.to(hidden_emo.device) 
        emo_out_emb = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1)) # [b, n_e_out, dim]
        emo_out_prob = torch.log(emo_out_prob.suqeeze(1))
        return emo_out_emb, mu, logvar, emo_out_prob
    
    def forward_train(self,  hidden_emo, p_emo_in, p_strat, p_emo_out):
        b = emo_logits.size(0)
        mus, logvars = self.posterior(hidden_emo, p_emo_in, p_emo_out)
        zs = torch.zeros((b, self.latent_dim, self.n_strat)).to(hidden_emo.device)
        E_probs = torch.zeros((b, self.n_emo_out, self.n_strat)).to(hidden_emo.device) #[b, n_e_out, n_s]
        for i, (mu, logvar) in enumerate(zip(mus, logvars)):
            z = self.reparameterize(mu, logvar)
            zs[:,:,i] = z
            E_prob = torch.softmax(self.Dense_z_prior_positive(z), dim=-1)
            E_probs[:,:,i] = E_prob
        emo_out_prob = torch.bmm(E_probs, p_strat.unsqueeze(-1)).permute(0,2,1) #[b, n_e_out, n_s] [b,  n_s,1]  -> [b, n_e_out, 1] ->[b, 1, n_e_out]
        emotion_id = self.emotion_id.to(hidden_emo.device) 
        emo_out_emb = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1)) # [b, n_e_out, dim]
        emo_out_prob = torch.log(emo_out_prob.suqeeze(1))
        return emo_out_emb, mu, logvar, emo_out_prob
    @staticmethod
    def kl_div(mu_posterior, logvar_posterior, mu_prior=None, logvar_prior=None):
        """
        This code is adapted from:
        https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/83ca9dd96272d3a38978a1dfa316d06d5d6a7c77/model/utils/probability.py#L20
        """
        one = torch.FloatTensor([1.0]).to(mu_posterior.device)
        if mu_prior == None:
            mu_prior = torch.FloatTensor([0.0]).to(mu_posterior.device)
            logvar_prior = torch.FloatTensor([0.0]).to(logvar_posterior.device)
        kl_div = torch.sum(0.5 * (logvar_prior - logvar_posterior + (logvar_posterior.exp()+(mu_posterior-mu_prior).pow(2))/logvar_prior.exp() - one) )
        return kl_div
    
if __name__ == "__main__":
    n_emo_in = 3
    n_emo_out = 4
    n_strat = 5
    batch_size = 2
    tran = EmoTrans(n_emo_in, n_emo_out, n_strat, embed_dim = 32)
    emo_logits = torch.full((batch_size, n_emo_in), 3.1)
    strat_logits = torch.full((batch_size, n_strat), 2.2)
    emo_embed, emo_out_logits = tran(emo_logits, strat_logits)