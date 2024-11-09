from src.transformers import AutoTokenizer

tokenizer_1 = AutoTokenizer.from_pretrained("facebook/bart-base")
tokenizer_2 = AutoTokenizer.from_pretrained("facebook/bart-large")
print(len(tokenizer_1))
print(len(tokenizer_2))
print(tokenizer_1("I saw it in your bag."))
print(tokenizer_2("I saw it in your bag."))