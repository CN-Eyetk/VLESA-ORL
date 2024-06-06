from chatbot.agent import Chatbot

cb = Chatbot(model_path = "/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae4-wo_comet-ct0.2-svae-lc-je-tppm602/bleu2/epoch0_step39_2024-06-03/lr_1e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmix_rec_llama_loadtemp")
chat = [
    {"role":"user","content":"Hello!"},
    {"role":"assistant", "content":"Hi! What can I do for you today?"},
   #{"role":"user", "content":"I am feeling so sad. I failed in my final exam. I tried hard."}
]
for round in range(5):
    
    usr_content = input("please say something")
    usr_utt = {"role":"user", "content":usr_content}
    chat.append(usr_utt)
    resp = cb.response(chat)
    sys_utt = {"role":"assistant", "content":resp}
    print(sys_utt)
    chat.append(sys_utt)
    
