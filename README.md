The official repository of "Be Helpful but Don't Talk too Much-Enhancing Helpfulness in Conversations through Relevance in Multi-Turn Emotional Support"
(0)Before everything, it is highly recommended to install the environment through "empchat.yml"


(1)To reproduce, please run train.sh. But remember to download the helpfulness model and change the paths in "paras.yaml" to your local path (The second row is not necessary if you are not interested in how dialogGPT performs in ORL).

Here is the helpfulmodel:
https://connectpolyu-my.sharepoint.com/:u:/g/personal/22038459r_connect_polyu_hk/EdQrYc9w2t9Jpga03dFRSakB1LLRiHctqse8FWolsI-hVA?e=TveZZJ

(2)To interact with a checkpoint, please run interact.sh


(3)Here are some checkpoints available for interact:
The SFT one without before optimal relevance learning (ORL):
https://connectpolyu-my.sharepoint.com/:u:/g/personal/22038459r_connect_polyu_hk/EWwbzpjCEYxJvcMuqGJqjn0BZ0MjehWM65ZxnOpu8WyOQA?e=pwyvnb


The RL one after optimal relevance learning (ORL):
https://connectpolyu-my.sharepoint.com/:u:/g/personal/22038459r_connect_polyu_hk/ERPuQOt0A3tMr420NCfDwa0BQK4onB1C_17BPK5Ag6qh9g?e=Nspw2L

The RL one after optimal relevance learning, but without learning from predicted processing load:
https://connectpolyu-my.sharepoint.com/:u:/g/personal/22038459r_connect_polyu_hk/EXjRMVhK8dlNsdQyGujyYRIBkK051rfq4gsk5e_MRBL5JQ?e=j379PS



