from rewarder import LLamaSeekerAgent
model_name = "/disk/junlin/EmoSp/llama/checkpoint-200"
agent = LLamaSeekerAgent(model_name)
contents_A =  [{"content":"Hello! What can I do for you today?","speaker":"supporter"},
                {"content":"My partner cheated on me.","speaker":"seeker"},
                {"content":"I am sorry to hear that.","speaker":"supporter"},
                {"content":"Can you tell me more about what happened?","speaker":"supporter"},
                ]
contents_B =  [{'content': "Hi, are you having a good day at the moment?", 'speaker': 'supporter'}, 
             {'content': "Today is okay, I guess. I' m just stressed.", 'speaker': 'seeker'}, 
             {'content': 'I am okay though, thanks for asking. Would you like to talk to me more about it?', 'speaker': 'supporter'},
            ]
print(agent.make_prompt(contents_B))
print(agent.response(contents_B))
