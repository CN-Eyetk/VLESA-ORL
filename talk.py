from chatbot.agent import Chatbot
import streamlit as st



dirs = ["/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae4-wo_comet-ct0.2-svae-lc-je-tppm608/bleu2/epoch0_step78_2024-06-03/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmix_rec_llama_load_1.5temp",
        "/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae4-wo_comet-ct0.2-svae-lc-je-tppm608/bleu2/epoch0_step78_2024-06-03/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmix_rec_llama_load_0.1temp",
        ]

cb_A = Chatbot(model_path = dirs[0])
cb_B = Chatbot(model_path = dirs[1])
chat = [
    
]



for i in range(5):
    prompt = input("Talk something")
    usr_utt = {"role":"user", "content":prompt}
    chat.append(usr_utt)
    resp_A = cb_A.response(chat)
    resp_B = cb_B.response(chat)
    sys_utt_A = {"role":"assistant", "content":resp_A}
    sys_utt_B = {"role":"assistant", "content":resp_B}
    print(sys_utt_A)
    print(sys_utt_B)
    chat.append(sys_utt_B)
