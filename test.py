from rewarder import ChatGPTScore, Retrive_DiagHist
from transformers import BartTokenizer
import json
from tqdm import tqdm
prompt = """You are presented with an incomplete segment of a conversation between two individuals, one of which (Speaker A) is seeking emotional support and another (Speaker B) is providing support. 
        The "Bob" has opened up to the other about his/her feelings, struggles, and challenges.
        Based on the :last" turn of the Jack, please assess the helpfulness of the "Jack".
        You have 7 options: Toxic; Very Bad; Bad; Average; Good; Very Good; Perfect
        
        
        Here's a breakdown
        0: Toxic - Harmful, even Toxic (Especailly when criticizing the )
        10: Very Bad - Cold and unhelpful, even harmful
        20: Bad - Indifferent, uncaring, slightly cold
        30: Average - Nothing harmful, but nothing helpful either
        40: Good - effective and helpful, but can be imporved
        50: Very Good - outstanding and goes above and beyond expectations.
        60: Perfect - Very Understanding, and Empathetic

        Here is an Example
        Jack : Hello how are you?
        Bob : hello im looking for someone to talk to
        Bob : im fine how are you
        Jack : I'm doing ok I'm glad you are good. Is it snowing by you?
        Jack : Merry Christmas!
        Bob : thats great and no its not snowing its very cold thow
        Bob : merry christmas to you also
        Jack : How can I help you today?
        Bob : im having some issues with friends not actually being friends
        Jack : I hear you are having trouble figuring out which friends are really your friends and which ones aren't. Is that about right?
        You may answer by: 50, Very Good, purposefully exploring Bob's experience

        Jack : Hello, what is life like for you at the moment?
        Bob : Infinitely complicated.
        Bob : Too many decisions. I don't know what to do.
        Jack : I am sorry to hear that but I am happy to listen and help you if I can
        Jack : what sort of things are you trying to decide?
        Bob : I don't even know where to start.
        Bob : I'm trying to decide if I can build trust with him again.
        Bob : He lied about everything.
        Jack : Well, let's try to take it one problem at a time so's not to get overwhelmed. What is your biggest problem at the moment?
        Jack : Ah, am I to take it that a relationship has recently ended?
        You may answer by: 30, Average, Exploring Bob's experience but not empathetic enough

        Bob : Hey there
        Bob : How are you?
        Jack : I AM FINE
        Jack : HOW IS YOUR SIDE ?
        Bob : I am ok, I'm having a hard time dealing with the pandemic though.
        Jack : Please how may i be of help
        Bob : I am having a hard time being motivated to do anything.
        Jack : I am sorry to hear that, bye.
        You may answer by: 10, Very Bad, Not meeting the Bob"s expectation of support
        """
scorer = ChatGPTScore(base_prompt="prompt")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
retriever = Retrive_DiagHist(tokenizer)
convs = json.load(open("dataset/ESConv.json", "r+"))
convs = [conv["dialog"] for conv in convs][:10]
read_utterances = []
step = 0

bar = tqdm(enumerate(convs))
for n,conv in bar:
    inner_bar = tqdm(range(len(conv)))
    for i in inner_bar:
        cur_conv = retriever.json_2_conv(conv[step:i+1])
        if conv[i]["speaker"] == "supporter":
            if len(read_utterances) == 0:
                reply = scorer.get_score(cur_conv)
            else:
                reply = scorer.get_score( "Here comes a new turn, you may update your judgement\n"+ cur_conv)
            step = i+1
            conv[i]["reply"] = reply
    with open(f"dataset/chatgpt_output/new_conv_{n}.json", "w+") as file:
        json.dump(conv, file, indent=2)
