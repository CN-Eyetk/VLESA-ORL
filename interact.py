
import sys

from chatbot.agent import Chatbot
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
import json


def command_line(cb):
    #cb = Chatbot(model_path)
    if cb.use_situ:
        situ = input("Please describe your situation:")
    else:
        situ = None
    chat = [
        {"role":"user","content":"Hello!"},
        {"role":"assistant", "content":"Hello! How's your day going. I am here to listen."},
    #{"role":"user", "content":"I am feeling so sad. I failed in my final exam, though I tried hard."}
    ]
    for i in range(5):
        usr_input = input("please say something:")
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

def use_stream_list(cb, situ, model_name):
    st.title(model_name)
    #cb = Chatbot(model_path)

            

    chat = [
        {"role":"user","content":"Hello!"},
        {"role":"assistant", "content":"Hello! How's your day going. I am here to listen."},
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
            resp = cb.response(st.session_state.messages, situ)
            #sys_utt = {"role":"assistant", "content":resp}
            #print(sys_utt)
            #chat.append(sys_utt)
            message_placeholder.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})

if __name__ == "__main__":
    path = sys.argv[1]
    cb = Chatbot(path)
    chat = command_line(cb)

