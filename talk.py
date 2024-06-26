from chatbot.agent import Chatbot
import streamlit as st
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



dirs = ["/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4/all_loss-1.0_0.1_0.1_510-spst-nokl-vae16-ct0.1-svae-lc-je-tp-stg_4pm619",
        "/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4/all_loss-1.0_0.1_0.1_510-spst-nokl-vae16-ct0.1-svae-lc-je-tp-situ-stg_4pm619",
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
    chat.append(sys_utt_A)
