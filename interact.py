from chatbot.agent import Chatbot
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
import json



def command_line(cb):
    #cb = Chatbot(model_path)
    if cb.use_situ:
        situ = input("Please describe your situation")
    else:
        situ = None
    chat = [
        {"role":"user","content":"Hello!"},
        {"role":"assistant", "content":"Hello, how has life been treating you lately ?"},
    #{"role":"user", "content":"I am feeling so sad. I failed in my final exam, though I tried hard."}
    ]
    for i in range(5):
        usr_input = input("please say something")
        if usr_input == "exit":
            chat.append({"situation":situ})
            return chat
        new_utt = {"role": "user", "content": usr_input}
        chat.append(new_utt)
        resp = cb.response(chat, situ)
        print("System", resp)
        new_utt = {"role": "assistant", "content": resp}
        chat.append(new_utt)
    chat.append({"situation":situ})
    return chat

def use_stream_list(model_path):
    cb = Chatbot(model_path)
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

if __name__ == "__main__":
    path_A = "/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4/all_loss-1.0_0.1_0.1_510-spst-nokl-vae16-ct0.1-svae-lc-je-tp-situ-stg_8am922"
    path_B = "/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.1_0.1_510-spst-nokl-vae16-ct0.1-svae-lc-je-tp-situ-stg_8am922/epoch0_step78_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmix_rec_llama_load_1.5temp"
    cb_A = Chatbot(path_A)
    cb_B = Chatbot(path_B)
    for i in range(8, 50):
        chat_A = command_line(cb_A)
        print("+++++++++++++++++++++++++++++++++++++++++++")
        chat_B = command_line(cb_B)
        print("===========================================")
        with open(f"outputs/base/chat_{i}.json","w+") as file:
            json.dump(chat_A, file, indent = 2)
        with open(f"outputs/ppo/chat_{i}.json","w+") as file:
            json.dump(chat_B, file, indent = 2)
