from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

# The `tokenizer` variable is being initialized with a tokenizer model loaded from the
# 'meta-llama/Meta-Llama-3.1-8B-Instruct' pretrained model using the Hugging Face Transformers library
# in Python. This tokenizer is used to preprocess input text data before feeding it into the model for
# natural language processing tasks.
tokenizer.apply_chat_template(
    [{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}], tokenize=False
)