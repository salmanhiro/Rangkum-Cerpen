from transformers import BertTokenizer, EncoderDecoderModel
import os
import streamlit as st


st.header('Rangkuman Cerpen')
st.text('powered by BERT')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

tokenizer = BertTokenizer.from_pretrained("cahya/bert2bert-indonesian-summarization")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
model = EncoderDecoderModel.from_pretrained("cahya/bert2bert-indonesian-summarization")

# 
ARTICLE_TO_SUMMARIZE = st.text_area("Masukkan cerpen yang ingin diringkas (max 512 token)")


# generate summary
input_ids = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors='pt')
summary_ids = model.generate(input_ids,
            min_length=20,
            max_length=80, 
            num_beams=10,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True,
            no_repeat_ngram_size=2,
            use_cache=True,
            do_sample = True,
            temperature = 0.8,
            top_k = 50,
            top_p = 0.95)

summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

st.write(summary_text)