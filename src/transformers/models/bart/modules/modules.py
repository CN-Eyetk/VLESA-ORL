import torch
import torch.nn.functional as F
from torch import nn
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
        self.matrices = nn.ParameterList([nn.Parameter(torch.Tensor(n_emo_in, n_emo_out)) for i in range(self.n_strat )])
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
        #for i in range(len(self.matrices)):
        #    with torch.no_grad():
        #        weight_norm = self.matrices[i]/self.matrices[i].sum(dim=1, keepdim=True)
        #        self.matrices[i].copy_(weight_norm)
        #    emo_out_logits_cur_strat = F.linear(emo_prob, self.matrices[i].t())
        #    emo_out_logits_each_strat[:, i, :] = emo_out_logits_cur_strat
        strat_prob = F.softmax(strat_logits, dim = -1)
        emo_out_prob = torch.bmm(strat_prob.unsqueeze(-2), emo_out_logits_each_strat) #[b, 1, stra] * [b, stra, emo] -> [b, 1, emo] 
        emotion_id = self.emotion_id.to(emo_logits.device) 
        emo_embed = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1))
        emo_out_prob = emo_out_prob.squeeze()
        emo_out_prob = torch.log(emo_out_prob) #upDATE  9-27-II
        return emo_embed, emo_out_prob
        

class EmoTrans_wo_STRA(nn.Module):
    def __init__(self, n_emo_in, n_emo_out, n_strat, embed_dim):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.n_emo_out = n_emo_out
        self.n_strat = 1
        self.embed_dim = embed_dim
        self.matrices = nn.ParameterList([nn.Parameter(torch.Tensor(n_emo_in, n_emo_out)) for i in range(self.n_strat )])
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
        #strat_prob = F.softmax(strat_logits, dim = -1)
        emo_out_prob =  emo_out_logits_each_strat #[b, 1, emo_out]
        emotion_id = self.emotion_id.to(emo_logits.device) 
        emo_embed = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1))
        emo_out_prob = emo_out_prob.squeeze()
        emo_out_prob = torch.log(emo_out_prob) #upDATE  9-27-II
        return emo_embed, emo_out_prob

class EmoTrans_wo_Emo(nn.Module):
    def __init__(self, n_emo_in, n_emo_out, n_strat, embed_dim):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.n_emo_out = n_emo_out
        self.n_strat = n_strat
        self.embed_dim = embed_dim
        self.matrix = nn.Parameter(torch.Tensor(n_strat, n_emo_out))
        self.emotion_embedding = nn.Embedding(n_emo_out, embed_dim)
        self.emotion_id = torch.tensor(range(n_emo_out), dtype=torch.long)
        self.dropout = nn.Dropout(0.1)
        self.reset_weights()
    def reset_weights(self):

        torch.nn.init.ones_(
                self.matrix)
    def forward(self, emo_logits, strat_logits):
        b = emo_logits.size(0)
        #emo_out_logits_each_strat = torch.zeros(b, self.n_strat, self.n_emo_out).to(emo_logits.device) #[b, stra, emo]
        #emo_logits = self.dropout(emo_logits)
        strat_logits = self.dropout(strat_logits)
        strat_prob = F.softmax(strat_logits, dim = -1)
        if len(strat_prob.size()) == 2:
            strat_prob = strat_prob.unsqueeze(-2)
        #emo_prob = F.softmax(emo_logits, dim = -1)
        with torch.no_grad():
            weight_norm = self.matrix/self.matrix.sum(dim=1, keepdim=True)
            self.matrix.copy_(weight_norm)
        emo_out_logits = F.linear(strat_prob, self.matrix.t())
            
        #strat_prob = F.softmax(strat_logits, dim = -1)
        emo_out_prob =  F.softmax(emo_out_logits, dim = -1) #[b, 1, emo_out]
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
        #self.batch_norm = nn.BatchNorm1d(n_hidden_out)
        #self.dropout = nn.Dropout(0.1)
    
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
        #print("hidden_targ_new",hidden_targ_new.shape)
        #print("hidden_targ_new", hidden_targ_new.shape)
        #print("hidden_src", hidden_src.shape)
        W_s_h = self.W(torch.cat((hidden_targ_new, hidden_src), dim=2))
        #W_s_h = self.batch_norm(W_s_h.permute(0,2,1)).permute(0,2,1)
        tanh_W_s_h = torch.tanh(W_s_h)  #[b, src_seq_len, n_hidden_dec]
        #print("tanh_W_s_h", tanh_W_s_h.shape)
        #tanh_W_s_h = self.dropout(tanh_W_s_h)
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)       #[b, n_hidde_dec, seq_len]
        #print("tanh_W_s_h",tanh_W_s_h.shape)
        
        V = self.V.repeat(batch_size, 1).unsqueeze(1)  #[b, 1, n_hidden_dec]
        #print("V",V.shape)
        e = torch.bmm(V, tanh_W_s_h).squeeze(1)        #[b, seq_len]
        #print("e",e.shape)
        mask[mask == 0] = -1e8
        e = e*mask
        att_weights = F.softmax(e, dim=1)              #[b, src_seq_len]
        #print("att_weights",att_weights.shape)
        #att_weights = att_weights * mask
        hidden_output = torch.bmm(att_weights.unsqueeze(1), hidden_src).squeeze(1)        
        assert hidden_output.size() == hidden_targ.size()
        return att_weights, hidden_output


class IntensityVAE(nn.Module):
    def __init__(self, config, n_emo_in, n_strat):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.out_dim = 3
        self.n_strat = n_strat
        
        self.hidden_dim = config.d_model + self.n_emo_in
        self.hidden_dim_encoder = self.hidden_dim + self.out_dim
        self.latent_dim = config.latent_dim
        
        self.h_prior = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.mu_prior = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar_prior = nn.Linear(self.hidden_dim, self.latent_dim)
        self.Dense_z_prior = nn.Linear(self.latent_dim, self.out_dim, bias = True)
        
        self.h_posterior = nn.Linear(self.hidden_dim_encoder, self.hidden_dim_encoder)
        self.mu_posterior = nn.Linear(self.hidden_dim_encoder, self.latent_dim)
        self.logvar_posterior = nn.Linear(self.hidden_dim_encoder, self.latent_dim)
        self.project = nn.Linear(self.latent_dim, config.d_model, bias = False)
        

    def prior(self,  hidden_prior):
        h1 = F.elu(self.h_prior(hidden_prior))
        mu = self.mu_prior(h1)
        logvar = self.logvar_prior(h1)
        return mu, logvar
    
    def posterior(self, hidden_prior, intensity):
        h1 = F.elu(self.h_posterior(torch.cat((hidden_prior, intensity), dim = -1)))
        mu = self.mu_posterior(h1)
        logvar = self.logvar_posterior(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, hidden_prior):
        mu, logvar = self.prior(hidden_prior)
        z = self.reparameterize(mu, logvar)
        intensity = torch.sigmoid(self.Dense_z_prior(z))
        emo_out_emb = self.project(z)
        return emo_out_emb, mu, logvar, intensity
    
    def forward_train(self,  hidden_prior, intensity):
        mu, logvar = self.posterior(hidden_prior, intensity)
        z = self.reparameterize(mu, logvar)
        intensity_reconst = torch.sigmoid(self.Dense_z_prior(z)) #[b, 1,n_emo]
        emo_out_emb = self.project(z)
        return emo_out_emb, mu, logvar, intensity_reconst
    @staticmethod
    def kl_div(mu_posterior, logvar_posterior, mu_prior=None, logvar_prior=None):
        """
        This code is adapted from:
        https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/83ca9dd96272d3a38978a1dfa316d06d5d6a7c77/model/utils/probability.py#L20
        """
        
        one = torch.FloatTensor([1.0]).to(mu_posterior.device)
        b = mu_posterior.size(0)
        
        if mu_prior == None:
            mu_prior = torch.FloatTensor([0.0]).to(mu_posterior.device)
            logvar_prior = torch.FloatTensor([0.0]).to(logvar_posterior.device)
        kl_div = torch.sum(0.5 * (logvar_prior - logvar_posterior + (logvar_posterior.exp()+(mu_posterior-mu_prior).pow(2))/logvar_prior.exp() - one) )
        kl_div = kl_div / b
        return kl_div
    
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
        #self.Dense_z_posterior = nn.Linear(self.latent_dim, self.n_emo_out)
        

    def prior(self,  hidden_prior):
        h1 = F.relu(self.h_prior(hidden_prior))
        mu = self.mu_prior(h1)
        logvar = self.logvar_prior(h1)
        return mu, logvar
    
    def posterior(self, hidden_prior, emo_out_emb):
        h1 = F.relu(self.h_posterior(torch.cat((hidden_prior, emo_out_emb), dim = -1)))
        mu = self.mu_posterior(h1)
        logvar = self.logvar_posterior(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, hidden_prior, p_emo_in, p_strat, emo_out_emb):
        p_emo_in = self.dropout(p_emo_in)
        p_emo_in = F.softmax(p_emo_in)
        p_strat = self.dropout(p_strat)
        p_strat = F.softmax(p_strat)
        b = p_emo_in.size(0)
        mu, logvar = self.prior( hidden_prior, p_emo_in, p_strat)
        z = self.reparameterize(mu, logvar)
        E_prob = torch.softmax(self.Dense_z_prior(z), dim=-1).unsqueeze(-2) #[b, 1,n_emo]
        emotion_id = self.emotion_id.to(hidden_prior.device) 
        emo_out_emb = torch.bmm(E_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1))
        E_prob = torch.log(E_prob.squeeze(1))
        return emo_out_emb, mu, logvar, E_prob
    
    def forward_train(self,  hidden_emo, hidden_strat, p_emo_in, p_strat, p_emo_out):
        p_emo_in = self.dropout(p_emo_in)
        p_emo_in = F.softmax(p_emo_in)
        p_strat = self.dropout(p_strat)
        p_strat = F.softmax(p_strat)
        b = p_emo_in.size(0)
        mu, logvar = self.posterior(hidden_emo, hidden_strat, p_emo_in, p_strat, p_emo_out)
        z = self.reparameterize(mu, logvar)
        E_prob = torch.softmax(self.Dense_z_prior(z), dim=-1).unsqueeze(-2) #[b, 1,n_emo]
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
        b = mu_posterior.size(0)
        
        if mu_prior == None:
            mu_prior = torch.FloatTensor([0.0]).to(mu_posterior.device)
            logvar_prior = torch.FloatTensor([0.0]).to(logvar_posterior.device)
        kl_div = torch.sum(0.5 * (logvar_prior - logvar_posterior + (logvar_posterior.exp()+(mu_posterior-mu_prior).pow(2))/logvar_prior.exp() - one) )
        kl_div = kl_div / b
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
        self.hidden_dim_prior = self.hidden_dim
        if config.sample_strat_emb:
            self.hidden_dim_prior += config.d_model
        if config.emo_use_cat_attn:
            self.hidden_dim_prior += config.d_model
        self.hidden_dim_post= self.hidden_dim
        self.latent_dim = config.latent_dim
        
        self.h_prior_emo = nn.Linear(self.hidden_dim_prior,self.hidden_dim_prior)
        self.mu_priors = nn.ModuleList([nn.Linear(self.hidden_dim_prior, self.latent_dim)  for i in range(self.n_strat)])
        self.logvar_priors = nn.ModuleList([nn.Linear(self.hidden_dim_prior, self.latent_dim)  for i in range(self.n_strat)])
        self.Dense_z_priors =  nn.ModuleList([nn.Linear(self.latent_dim, self.n_emo_out)  for i in range(self.n_strat)])


        self.hidden_dim_pos = self.hidden_dim_prior + self.hidden_dim
        self.h_posterior_emo = nn.Linear(self.hidden_dim_pos, self.hidden_dim_pos)
        #self.h_posterior_strat = nn.Linear(self.hidden_dim + self.n_strat + self.n_emo_out, self.latent_dim)
        self.mu_posteriors = nn.ModuleList([nn.Linear(self.hidden_dim_pos, self.latent_dim)  for i in range(self.n_strat)])
        self.logvar_posteriors = nn.ModuleList([nn.Linear(self.hidden_dim_pos, self.latent_dim)  for i in range(self.n_strat)])
        #self.Dense_z_posterior = nn.Linear(self.latent_dim, self.n_emo_out) 
        #self.z_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        

    def prior(self, hidden_prior):
        b = hidden_prior.size(0)
        h1_emo = F.relu(self.h_prior_emo(hidden_prior))
        mus = torch.zeros((b, self.latent_dim, self.n_strat)).to(hidden_prior.device)
        logvars = torch.zeros((b, self.latent_dim, self.n_strat)).to(hidden_prior.device)
        for i in range(len(self.mu_priors)):
            #print("h1_emo",h1_emo.shape)
            mu = self.mu_priors[i](h1_emo)
            logvar = self.logvar_priors[i](h1_emo)
            mus[:,:,i] = mu
            logvars[:, :, i] = logvar
        
        return mus, logvars
    
    def posterior(self, hidden_prior, hidden_post):
        b = hidden_prior.size(0)
        #print("hidden_emo",hidden_emo.dtype)
        #print("p_emo_in",p_emo_in.dtype)
        #print("p_emo_out",p_emo_out.dtype)
        h1_emo = F.relu(self.h_posterior_emo(torch.cat((hidden_prior, hidden_post), dim = -1)))
        #h1_strat = F.relu(self.h_posterior_strat(torch.cat((hidden_strat, p_strat, p_emo_out), dim = -1)))
        mus = torch.zeros((b, self.latent_dim, self.n_strat)).to(h1_emo.device)
        logvars = torch.zeros((b, self.latent_dim, self.n_strat)).to(h1_emo.device)
        for i in range(len(self.mu_priors)):
            
            mu = self.mu_posteriors[i](h1_emo)
            logvar = self.logvar_posteriors[i](h1_emo)
            mus[:,:,i] = mu
            logvars[:, :, i] = logvar
        return mus, logvars

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(logvar.device)
        eps = torch.randn_like(std).to(logvar.device)
        return mu + eps * std

    def forward(self, hidden_prior, p_emo_in, p_strat):
        b = p_emo_in.size(0)
        #p_emo_in = self.dropout(p_emo_in)
        #p_emo_in = F.softmax(p_emo_in)
        #p_strat = self.dropout(p_strat)
        #p_strat = F.softmax(p_strat)
        mus, logvars = self.prior(hidden_prior)
        zs = torch.zeros((b, self.latent_dim, self.n_strat)).to(hidden_prior.device)
        E_probs = torch.zeros((b, self.n_emo_out, self.n_strat)).to(hidden_prior.device) #[b, n_e_out, n_s]
        for i in range(self.n_strat):
            z = self.reparameterize(mus[:,:,i], logvars[:,:,i])
            z = z.to(hidden_prior.device)
            
            #print("z",z.shape)
            zs[:,:,i] = z
            E_prob = torch.softmax(self.Dense_z_priors[i](z), dim=-1)
            E_probs[:,:,i] = E_prob
        emo_out_prob = torch.bmm(E_probs, p_strat.unsqueeze(-1)).permute(0,2,1) #[b, n_e_out, n_s] [b,  n_s,1]  -> [b, n_e_out, 1] ->[b, 1, n_e_out]
        #z_out = torch.bmm(zs, p_strat.unsqueeze(-1)).permute(0,2,1).squeeze(-2)
        emotion_id = self.emotion_id.to(hidden_prior.device) 
        #emo_out_emb = self.z_proj(z_out)
        emo_out_emb = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1)) # [b, n_e_out, dim]
        #emo_out_emb = emo_out_emb.unsqueeze(-2)
        emo_out_prob = torch.log(emo_out_prob.squeeze(1))
        #print("emo_out_emb",emo_out_emb.shape)
        return emo_out_emb, mus, logvars, emo_out_prob
    
    def forward_train(self,  hidden_prior, p_emo_in, p_strat, hidden_post, q_0): #p_emo_in 和 p_strat输入的都是logits
        #p_emo_in = self.dropout(p_emo_in)
        #p_emo_in = F.softmax(p_emo_in)
        #p_strat = self.dropout(p_strat)
        #p_strat = F.softmax(p_strat)
        
        b = p_emo_in.size(0)
        mus, logvars = self.posterior(hidden_prior, hidden_post.to(hidden_prior.dtype))
        zs = torch.zeros((b, self.latent_dim, self.n_strat)).to(hidden_prior.device)
        E_probs = torch.zeros((b, self.n_emo_out, self.n_strat)).to(hidden_prior.device) #[b, n_e_out, n_s]
        for i in range(self.n_strat):
            z = self.reparameterize(mus[:,:,i], logvars[:,:,i])
            z = z.to(hidden_prior.device)
            zs[:,:,i] = z
            E_prob = torch.softmax(self.Dense_z_priors[i](z), dim=-1)
            E_probs[:,:,i] = E_prob
        emo_out_prob = torch.bmm(E_probs, p_strat.unsqueeze(-1)).permute(0,2,1) #[b, n_e_out, n_s] [b,  n_s,1]  -> [b, n_e_out, 1] ->[b, 1, n_e_out]
        emotion_id = self.emotion_id.to(hidden_prior.device) 
        emo_out_emb = torch.bmm(q_0.unsqueeze(-2).to(emo_out_prob.dtype),  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1)) # [b, n_e_out, dim]
        #z_out = torch.bmm(zs, p_strat.unsqueeze(-1)).permute(0,2,1).squeeze(-2)
        #emo_out_emb = self.z_proj(z_out)
        #emo_out_emb = emo_out_emb.unsqueeze(-2)
        emo_out_prob = torch.log(emo_out_prob.squeeze(1))
        #print("emo_out_emb",emo_out_emb.shape)
        return emo_out_emb, mus, logvars, emo_out_prob
    @staticmethod
    def kl_div(mu_posterior, logvar_posterior, mu_prior=None, logvar_prior=None):
        """
        This code is adapted from:
        https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/83ca9dd96272d3a38978a1dfa316d06d5d6a7c77/model/utils/probability.py#L20
        """
        b = mu_posterior.size(0)
        one = torch.FloatTensor([1.0]).to(mu_posterior.device)
        if mu_prior == None:
            mu_prior = torch.FloatTensor([0.0]).to(mu_posterior.device)
            logvar_prior = torch.FloatTensor([0.0]).to(logvar_posterior.device)
        kl_div = torch.sum(0.5 * (logvar_prior - logvar_posterior + (logvar_posterior.exp()+(mu_posterior-mu_prior).pow(2))/logvar_prior.exp() - one) )
        kl_div = kl_div  / b
        return kl_div


class EmoTransVAE_MixStrat(nn.Module):
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
        self.hidden_dim_prior_stg = self.hidden_dim
        self.hidden_dim_prior_emo = self.hidden_dim
        if config.stg_use_cat_attn:
            self.hidden_dim_prior_stg += config.d_model
        if config.emo_use_cat_attn:
            self.hidden_dim_prior_emo += config.d_model
        self.latent_dim = config.latent_dim
        
        self.h_prior_emo = nn.Linear(self.hidden_dim_prior_emo, self.hidden_dim_prior_emo)
        self.h_prior_strat = nn.Linear(self.hidden_dim_prior_stg, self.hidden_dim_prior_stg)
        self.mu_prior = nn.Linear(self.hidden_dim_prior_emo, self.latent_dim)
        self.logvar_prior = nn.Linear(self.hidden_dim_prior_stg, self.latent_dim)
        self.Dense_z_prior =  nn.Linear(self.latent_dim, self.n_emo_out)


        self.hidden_dim_post_emo = self.hidden_dim_prior_emo + self.hidden_dim
        self.hidden_dim_post_stg = self.hidden_dim_prior_stg + self.hidden_dim
        self.h_posterior_emo = nn.Linear(self.hidden_dim_post_emo, self.hidden_dim_post_emo)
        self.h_posterior_strat = nn.Linear(self.hidden_dim_post_stg, self.hidden_dim_post_stg)
        self.mu_posterior = nn.Linear(self.hidden_dim_post_emo, self.latent_dim)
        self.logvar_posterior = nn.Linear(self.hidden_dim_post_stg, self.latent_dim)

    def prior(self, hidden_prior_emo, hidden_prior_strat):
        b = hidden_prior_emo.size(0)
        h1_emo = F.relu(self.h_prior_emo(hidden_prior_emo))
        h1_strat = F.relu(self.h_prior_strat(hidden_prior_strat))
        mu_emo = self.mu_prior(h1_emo)
        logvar_strat = self.logvar_prior(h1_strat)
        return mu_emo, logvar_strat
    
    def posterior(self, hidden_post_emo, hidden_post_strat):
        b = hidden_post_emo.size(0)
        h1_emo = F.relu(self.h_posterior_emo(hidden_post_emo))
        h1_strat = F.relu(self.h_posterior_strat(hidden_post_strat))
        mu_emo = self.mu_posterior(h1_emo)
        logvar_strat = self.logvar_posterior(h1_strat)
        return mu_emo, logvar_strat

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(logvar.device)
        eps = torch.randn_like(std).to(logvar.device)
        return mu + eps * std

    def forward(self, hidden_prior_emo, hidden_prior_strat):
        b = hidden_prior_emo.size(0)
        mus, logvars = self.prior(hidden_prior_emo, hidden_prior_strat)
        z = self.reparameterize(mus, logvars)
        z = z.to(hidden_prior_emo.device)
        emo_out_prob = torch.softmax(self.Dense_z_prior(z), dim=-1).unsqueeze(-2)
        emotion_id = self.emotion_id.to(hidden_prior_emo.device) 
        #print("emo_out_prob",emo_out_prob.shape)
        emo_out_emb = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1)) # [b, n_e_out, dim]
        emo_out_prob = torch.log(emo_out_prob.squeeze(1))
        return emo_out_emb, mus, logvars, emo_out_prob
    
    def forward_train(self,  hidden_post_emo, hidden_post_strat, q_0): #p_emo_in 和 p_strat输入的都是logits
        b = hidden_post_emo.size(0)
        mus, logvars = self.posterior(hidden_post_emo, hidden_post_strat)
        z = self.reparameterize(mus, logvars)
        z = z.to(hidden_post_emo.device)
        emo_out_prob = torch.softmax(self.Dense_z_prior(z), dim=-1).unsqueeze(-2)
        emotion_id = self.emotion_id.to(emo_out_prob.device) 
        
        emo_out_emb = torch.bmm(q_0.unsqueeze(-2).to(emo_out_prob.dtype),  
                                self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1)) # [b, n_e_out, dim]
        emo_out_prob = torch.log(emo_out_prob.squeeze(1))
        return emo_out_emb, mus, logvars, emo_out_prob
    @staticmethod
    def kl_div(mu_posterior, logvar_posterior, mu_prior=None, logvar_prior=None):
        """
        This code is adapted from:
        https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/83ca9dd96272d3a38978a1dfa316d06d5d6a7c77/model/utils/probability.py#L20
        """
        b = mu_posterior.size(0)
        one = torch.FloatTensor([1.0]).to(mu_posterior.device)
        if mu_prior == None:
            mu_prior = torch.FloatTensor([0.0]).to(mu_posterior.device)
            logvar_prior = torch.FloatTensor([0.0]).to(logvar_posterior.device)
        kl_div = torch.sum(0.5 * (logvar_prior - logvar_posterior + (logvar_posterior.exp()+(mu_posterior-mu_prior).pow(2))/logvar_prior.exp() - one) )
        kl_div = kl_div  / b
        return kl_div

class EmoTransVAE_MultiStrat_Light(nn.Module):
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
        self.hidden_dim_prior = self.hidden_dim
        if config.sample_strat_emb:
            self.hidden_dim_prior += config.d_model
        if config.use_cat_attn:
            self.hidden_dim_prior += config.d_model
        self.hidden_dim_post= self.hidden_dim
        self.latent_dim = int(self.hidden_dim/2)
        
        self.h_prior_emo = nn.Linear(self.hidden_dim_prior, self.latent_dim)
        self.mu_prior = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_prior = nn.Linear(self.latent_dim, self.latent_dim)
        self.Dense_z_prior = nn.Linear(self.latent_dim, self.n_emo_out)

        self.hidden_dim_pos = self.hidden_dim_prior+self.hidden_dim
        self.h_posterior_emo = nn.Linear(self.hidden_dim_pos, self.latent_dim)
        #self.h_posterior_strat = nn.Linear(self.hidden_dim + self.n_strat + self.n_emo_out, self.latent_dim)
        self.mu_posterior = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_posterior = nn.Linear(self.latent_dim, self.latent_dim)
        #self.Dense_z_posterior = nn.Linear(self.latent_dim, self.n_emo_out) 
        

    def prior(self, hidden_prior):
        b = hidden_prior.size(0)
        h1_emo = F.relu(self.h_prior_emo(hidden_prior))
        mu = self.mu_prior(h1_emo).to(h1_emo.device)
        logvar = self.logvar_prior(h1_emo).to(h1_emo.device)
        return mu, logvar
    
    def posterior(self, hidden_prior, hidden_post):
        b = hidden_prior.size(0)
        h1_emo = F.relu(self.h_posterior_emo(torch.cat((hidden_prior, hidden_post), dim = -1)))
        mu = self.mu_posterior(h1_emo).to(h1_emo.device) 
        logvar = self.logvar_posterior(h1_emo).to(h1_emo.device) 
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(logvar.device)
        eps = torch.randn_like(std).to(logvar.device)
        return mu + eps * std

    def forward(self, hidden_prior, p_emo_in, p_strat):
        b = hidden_prior.size(0)
        p_strat = self.dropout(p_strat)
        p_strat = F.softmax(p_strat)
        mu, logvar = self.prior(hidden_prior)
        z = self.reparameterize(mu, logvar)
        z = z.to(hidden_prior.device)
        emo_out_prob = torch.softmax(self.Dense_z_prior(z), dim=-1)
        
        #emo_out_prob = torch.bmm(E_probs, p_strat.unsqueeze(-1)).permute(0,2,1) #[b, n_e_out, n_s] [b,  n_s,1]  -> [b, n_e_out, 1] ->[b, 1, n_e_out]
        emotion_id = self.emotion_id.to(hidden_prior.device) 
        if len(emo_out_prob.size() == 2):
            emo_out_prob = emo_out_prob.unsqueeze(-2)
        emo_out_emb = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1)) # [b, n_e_out, dim]
        emo_out_prob = torch.log(emo_out_prob.squeeze(1))
        return emo_out_emb, mu, logvar, emo_out_prob
    
    def forward_train(self,  hidden_prior, p_emo_in, p_strat, hidden_post): 

        b = hidden_prior.size(0)
        mu, logvar = self.posterior(hidden_prior, hidden_post)
        z = self.reparameterize(mu, logvar)
        z = z.to(hidden_prior.device)
        emo_out_prob = torch.softmax(self.Dense_z_prior(z), dim=-1)
        #emo_out_prob = torch.bmm(E_probs, p_strat.unsqueeze(-1)).permute(0,2,1) #[b, n_e_out, n_s] [b,  n_s,1]  -> [b, n_e_out, 1] ->[b, 1, n_e_out]
        emotion_id = self.emotion_id.to(hidden_prior.device) 
        if len(emo_out_prob.size() == 2):
            emo_out_prob = emo_out_prob.unsqueeze(-2)
        emo_out_emb = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1)) # [b, n_e_out, dim]
        emo_out_prob = torch.log(emo_out_prob.squeeze(1))
        return emo_out_emb,  mu, logvar, emo_out_prob
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

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        #print("labels",labels)
        #print("features",features.shape)
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if torch.isinf(features).any() or torch.isnan(features).any():
            print("features has nan")
            assert 1 == 2
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        mean_log_prob_pos[mean_log_prob_pos != mean_log_prob_pos] = 0.0
        print("mean_log_prob_pos",mean_log_prob_pos)
        print("mask_pos_pairs",mask_pos_pairs)
        if torch.isinf(mean_log_prob_pos).any() or torch.isnan(mean_log_prob_pos).any():
            print("log prob has nan")
            assert 1 == 2
            
        #    mean_log_prob_pos = torch.clamp(mean_log_prob_pos, max = 0.0)
        # loss
        
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        if torch.isinf(loss).any() or torch.isnan(loss).any():
            print("loss has nan")
            assert 1 == 2
        return loss

if __name__ == "__main__":
    n_emo_in = 3
    n_emo_out = 4
    n_strat = 5
    batch_size = 2
    tran = EmoTrans(n_emo_in, n_emo_out, n_strat, embed_dim = 32)
    emo_logits = torch.full((batch_size, n_emo_in), 3.1)
    strat_logits = torch.full((batch_size, n_strat), 2.2)
    emo_embed, emo_out_logits = tran(emo_logits, strat_logits)