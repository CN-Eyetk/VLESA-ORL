from chatbot.agent import Chatbot
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate


path_A = "/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4/all_loss-1.0_0.1_0.1_510-spst-nokl-vae16-ct0.1-svae-lc-je-tp-stg_8am922"
path_B = "/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.1_0.1_510-spst-nokl-vae16-ct0.1-svae-lc-je-tp-stg_8am922/epoch0_step78_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmix_rec_llama_load_1.5temp"


cb = Chatbot(model_path = path_B)
chat = [
    {"role":"user","content":"Hello!"},
    {"role":"assistant", "content":"Hi! How's your day?"},
   #{"role":"user", "content":"I am feeling so sad. I failed in my final exam, though I tried hard."}
]

if "messages" not in st.session_state:
    st.session_state.messages = chat
        
#with st.chat_message("user"):
#    st.markdown(chat[0]["content"])
    
#with st.chat_message("assistant"):
#    st.markdown(chat[1]["content"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    if prompt == "stop":
        st.session_state.messages = chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input("What is up?")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        #usr_utt = {"role":"user", "content":prompt}
        #chat.append(usr_utt)
        resp = cb.response(st.session_state.messages)
        #sys_utt = {"role":"assistant", "content":resp}
        #print(sys_utt)
        #chat.append(sys_utt)
        message_placeholder.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
