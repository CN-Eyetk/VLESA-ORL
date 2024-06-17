import json
import math
import glob
import operator
import spacy,en_core_web_sm

nlp = spacy.load("en_core_web_md")

def extract_vp(sentence):

    
    doc = nlp(sentence)
    tok_l = doc.to_json()['tokens']
    res = []
    for t in tok_l:
        head = tok_l[t['head']]
        res.append(f"'{sentence[t['start']:t['end']]}' is {t['dep']} of '{sentence[head['start']:head['end']]}'")
    return res
def head(stat_dict,hits = 20,hsort = True,output = False,filename = None, sep = "\t"):
	#first, create sorted list. Presumes that operator has been imported
	sorted_list = sorted(stat_dict.items(),key=operator.itemgetter(1),reverse = hsort)[:hits]

	if output == False and filename == None: #if we aren't writing a file or returning a list
		for x in sorted_list: #iterate through the output
			print(x[0] + "\t" + str(x[1])) #print the sorted list in a nice format

	elif filename is not None: #if a filename was provided
		outf = open(filename,"w") #create a blank file in the working directory using the filename
		outf.write("item\tstatistic") #write header
		for x in sorted_list: #iterate through list
			outf.write("\n" + x[0] + sep + str(x[1])) #write each line to a file using the separator
		outf.flush() #flush the file buffer
		outf.close() #close the file

	if output == True: #if output is true
		return(sorted_list) #return the sorted list
def ngrammer(token_list, gram_size, separator = " "):
	ngrammed = [] #empty list for n-grams

	for idx, x in enumerate(token_list): #iterate through the token list using enumerate()

		ngram = token_list[idx:idx+gram_size] #get current word token_list plus words n-words after (this is a list)

		if len(ngram) == gram_size: #don't include shorter ngrams that we would get at the end of a text
			ngrammed.append(separator.join(ngram)) # join the list of ngram items using the separator (by default this is a space), add to ngrammed list

	return(ngrammed) #return list of ngrams

def new_ngrammer(token_list, gram_size, separator = " "):
    ngrammed = [] #empty list for n-grams
    sent = " ".join(x for x in token_list)
    new_token_list = extract_vp(sent)
    return new_token_list

def tokenize(input_string,gram_size=1, separator = " "): #input_string = text string

	tokenized = [] #empty list that will be returned

	#these are the punctuation marks in the Brown corpus + '"'
	punct_list = ['-',',','.',"'",'&','`','?','!',';',':','(',')','$','/','%','*','+','[',']','{','}','"']

	#this is a sample (but potentially incomplete) list of items to replace with spaces
	replace_list = ["\n","\t"]

	#This is a sample (but potentially incomplete) list if items to ignore
	ignore_list = [""]

	#iterate through the punctuation list and delete each item
	for x in punct_list:
		input_string = input_string.replace(x, "") #instead of adding a space before punctuation marks, we will delete them (by replacing with nothing)

	#iterate through the replace list and replace it with a space
	for x in replace_list:
		input_string = input_string.replace(x," ")

	#our examples will be in English, so for now we will lower them
	#this is, of course optional
	input_string = input_string.lower()

	#then we split the string into a list
	input_list = input_string.split(" ")

	for x in input_list:
		if x not in ignore_list: #if item is not in the ignore list
			tokenized.append(x) #add it to the list "tokenized"

	if gram_size == 1: #if we are looking at single words, simply return tokenized
		return(tokenized)

	else: #otherwise, return n-gram text, using the ngrammer() function
		return(new_ngrammer(tokenized,gram_size,separator))

def write_corpus():
    test_file = "/home/lijunlin/lijunlin/ESCONV_ACL/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae8-wo_comet-ct0.2-svae-lc-je-tppm613/bleu2/non_mix/hyp_strategy.json"
    sents = json.load(open(test_file, "r+"))
    with open("corpus/A.txt","w+") as file:
        for i,sent in enumerate(sents):
            file.write(sent)
            if i < len(sent) - 1:
                file.write("\n")
                

def corpus_freq(dir_name,gram_size = 1,separator = " ", ):
	freq = {} #create an empty dictionary to store the word : frequency pairs

	#create a list that includes all files in the dir_name folder that end in ".txt"
	lines = json.load(open(dir_name, "r+"))

	#iterate through each file:
	for line in lines:
		#open the file as a string
		text = line.strip()
		#tokenize text using our tokenize() function
		tokenized = tokenize(text,gram_size,separator) #use tokenizer indicated in function argument (e.g., "tokenize()" or "ngramizer()")

		#iterate through the tokenized text and add words to the frequency dictionary
		for x in tokenized:
			#the first time we see a particular word we create a key:value pair
			if x not in freq:
				freq[x] = 1
			#when we see a word subsequent times, we add (+=) one to the frequency count
			else:
				freq[x] += 1

	return(freq) #return frequency dictionary

def keyness(freq_dict1,freq_dict2): #this assumes that raw frequencies were used. effect options = "log-ratio", "%diff", "odds-ratio"
	keyness_dict = {"log-ratio": {},"%diff" : {},"odds-ratio" : {}, "c1_only" : {}, "c2_only":{}}

	#first, we need to determine the size of our corpora:
	size1 = sum(freq_dict1.values()) #calculate corpus size by adding all of the values in the frequency dictionary
	size2 = sum(freq_dict2.values()) #calculate corpus size by adding all of the values in the frequency dictionary

	#How to calculate three measures of keyness:
	def log_ratio(freq1,size1,freq2,size2):  #see Gabrielatos (2018); Hardie (2014)
		freq1_norm = freq1/size1 * 1000000 #norm per million words
		freq2_norm = freq2/size2 * 1000000 #norm per million words
		index = math.log2(freq1_norm/freq2_norm) #calculate log ratio
		return(index)

	def perc_diff(freq1,size1,freq2,size2): #see Gabrielatos (2018); Gabrielatos and Marchi (2011)
		freq1_norm = freq1/size1 * 1000000 #norm per million words
		freq2_norm = freq2/size2 * 1000000 #norm per million words
		index = ((freq1_norm-freq2_norm) * 100)/freq2_norm #calculate perc_diff
		return(index)

	def odds_ratio(freq1,size1,freq2,size2): #see Gabrielatos (2018); Everitt (2002)
		index = (freq1/(size1-freq1))/(freq2/(size2-freq2))
		return(index)


	#make a list that combines the keys from each frequency dictionary:
	all_words = set(list(freq_dict1.keys()) + list(freq_dict2.keys())) #set() creates a set object that includes only unique items

	#if our items only occur in one corpus, we will add them to our "c1_only" or "c2_only" dictionaries, and then ignore them
	for item in all_words:
		if item not in freq_dict1:
			keyness_dict["c2_only"][item] = freq_dict2[item]/size2 * 1000000 #add normalized frequency (per million words) to c2_only dictionary
			continue #move to next item in the list
		if item not in freq_dict2:
			keyness_dict["c1_only"][item] = freq_dict1[item]/size1 * 1000000 #add normalized frequency (per million words) to c1_only dictionary
			continue #move to next item on the list

		keyness_dict["log-ratio"][item] = log_ratio(freq_dict1[item],size1,freq_dict2[item],size2) #calculate keyness using log-ratio

		keyness_dict["%diff"][item] = perc_diff(freq_dict1[item],size1,freq_dict2[item],size2) #calculate keyness using %diff

		keyness_dict["odds-ratio"][item] = odds_ratio(freq_dict1[item],size1,freq_dict2[item],size2) #calculate keyness using odds-ratio

	return(keyness_dict) #return dictionary of dictionaries



A_freq = corpus_freq("/home/lijunlin/lijunlin/ESCONV_ACL/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae8-wo_comet-ct0.2-svae-lc-je-tppm613/bleu2/non_mix/hyp_strategy.json", gram_size = 4)

B_freq = corpus_freq("/home/lijunlin/lijunlin/ESCONV_ACL/our_generated_data/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae4-wo_comet-ct0.2-svae-lc-je-tppm608/bleu2/epoch0_step78_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmix_rec_llama_load_1.5temp/non_mix/hyp_strategy.json", gram_size = 4)

brown_key_news_fic = keyness(A_freq, B_freq) #this will include all of our keyness dictionaries. Note that this is directional (if we switch the frequency dictionaries we will get different but complementary results)
head(brown_key_news_fic["c1_only"],20) #items that only occur in the newspaper corpus (the first frequency list we entered into the keyness() function)
print("=============")
head(brown_key_news_fic["c2_only"],20) #items that only occur in the newspaper corpus (the first frequency list we entered into the keyness() function)

