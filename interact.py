from chatbot.agent import Chatbot
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate





cb = Chatbot(model_path = "/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae8-wo_comet-ct0.2-svae-lc-je-tppm613/bleu2/")
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        usr_utt = {"role":"user", "content":prompt}
        chat.append(usr_utt)
        resp = cb.response(chat)
        sys_utt = {"role":"assistant", "content":resp}
        print(sys_utt)
        chat.append(sys_utt)
        message_placeholder.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
