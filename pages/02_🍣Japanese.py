from jisho_api.word import Word
from jisho_api.sentence import Sentence
import pandas as pd
import re
import requests 
import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
#from spacy.language import Language
from spacy.tokens import Doc
import spacy_ke
import streamlit as st

# Global variables
JA_TEXT = """（朝日新聞）台湾気分のパワースポット ＪＲ大久保駅南口のすぐそばにある「東京媽祖廟（まそびょう）」は、台湾で広く信仰されている道教の神様を祭る。居酒屋やコンビニが並ぶ通りで、金色の竜など豪華な装飾が施された４階建ての赤い建物はとても目立つ。"""
DESCRIPTION = "AI模型輔助語言學習：日語"
TOK_SEP = " | "

# External API callers
def parse_jisho_senses(word):
    res = Word.request(word)
    response = res.dict()
    if response["meta"]["status"] == 200:
        data = response["data"]
        commons = [d for d in data if d["is_common"]]
        if commons:
            common = commons[0] # Only get the first entry that is common
            senses = common["senses"]
            if len(senses) > 3:
                senses = senses[:3]
            with st.container():
                for idx, sense in enumerate(senses):
                    eng_def = "; ".join(sense["english_definitions"])
                    pos = "/".join(sense["parts_of_speech"])
                    st.write(f"Sense {idx+1}: {eng_def} ({pos})")
        else:
            st.info("Found no common words on Jisho!")
    else:
        st.error("Can't get response from Jisho!")


def parse_jisho_sentences(word):
    res = Sentence.request(word)
    try:
        response = res.dict()
        data = response["data"]
        if len(data) > 3:
            sents = data[:3]
        else:
            sents = data
        with st.container():
            for idx, sent in enumerate(sents):
                eng = sent["en_translation"]
                jap = sent["japanese"]
                st.write(f"Sentence {idx+1}: {jap}")
                st.write(f"({eng})")
    except:
        st.info("Found no results on Jisho!")
    
# Utility functions
def create_jap_df(tokens):
    seen_texts = []
    filtered_tokens = []
    for tok in tokens:
        if tok.text not in seen_texts:
            filtered_tokens.append(tok)
            
    df = pd.DataFrame(
      {
          "單詞": [tok.text for tok in filtered_tokens],
          "發音": ["/".join(tok.morph.get("Reading")) for tok in filtered_tokens],
          "詞形變化": ["/".join(tok.morph.get("Inflection")) for tok in filtered_tokens],
          "原形": [tok.lemma_ for tok in filtered_tokens],
          #"正規形": [tok.norm_ for tok in verbs],
      }
    )
    st.dataframe(df)
    csv = df.to_csv().encode('utf-8')
    st.download_button(
      label="下載表格",
      data=csv,
      file_name='jap_forms.csv',
      )
          
def filter_tokens(doc):
    clean_tokens = [tok for tok in doc if tok.pos_ not in ["PUNCT", "SYM"]]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_email]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_url]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_num]
    clean_tokens = [tok for tok in clean_tokens if not tok.is_punct]
    clean_tokens = [tok for tok in clean_tokens if not tok.is_space]
    return clean_tokens

def create_kw_section(doc):
    st.markdown("## 關鍵詞分析") 
    kw_num = st.slider("請選擇關鍵詞數量", 1, 10, 3)
    kws2scores = {keyword: score for keyword, score in doc._.extract_keywords(n=kw_num)}
    kws2scores = sorted(kws2scores.items(), key=lambda x: x[1], reverse=True)
    count = 1
    for keyword, score in kws2scores: 
        rounded_score = round(score, 3)
        st.write(f"{count} >>> {keyword} ({rounded_score})")
        count += 1 
            
# Page setting
st.set_page_config(
    page_icon="🤠",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(f"# {DESCRIPTION}") 

# Load the selected model
nlp = spacy.load("ja_ginza")
nlp.add_pipe("yake") # keyword extraction
          
# Merge entity spans to tokens
# nlp.add_pipe("merge_entities") 

st.markdown("## 待分析文本")     
st.info("請在下面的文字框輸入文本並按下Ctrl + Enter以更新分析結果")
text = st.text_area("",  JA_TEXT, height=200)
doc = nlp(text)
st.markdown("---")
    
keywords_extraction = st.checkbox("關鍵詞分析", False)
analyzed_text = st.checkbox("分析後文本", True)
defs_examples = st.checkbox("單詞解釋與例句", True)
morphology = st.checkbox("詞形變化", False)
ner_viz = st.checkbox("命名實體", True)
tok_table = st.checkbox("斷詞特徵", False)

if keywords_extraction:
    create_kw_section(doc)

if analyzed_text:
    st.markdown("## 分析後文本") 
    for idx, sent in enumerate(doc.sents):
        clean_tokens = [tok for tok in sent if tok.pos_ not in ["PUNCT", "SYM"]]
        tokens_text = [tok.text for tok in clean_tokens]
        readings = ["/".join(tok.morph.get("Reading")) for tok in clean_tokens]
        display = [f"{text} [{reading}]" for text, reading in zip(tokens_text, readings)]
        if display:
          display_text = TOK_SEP.join(display)
          st.write(f"{idx+1} >>> {display_text}")
        else:
          st.write(f"{idx+1} >>> EMPTY LINE")  

if defs_examples:
    st.markdown("## 單詞解釋與例句")
    clean_tokens = filter_tokens(doc)
    alphanum_pattern = re.compile(r"[a-zA-Z0-9]")
    clean_lemmas = [tok.lemma_ for tok in clean_tokens if not alphanum_pattern.search(tok.lemma_)]
    vocab = list(set(clean_lemmas))
    if vocab:
        selected_words = st.multiselect("請選擇要查詢的單詞: ", vocab, vocab[0:3])
        for w in selected_words:
            st.write(f"### {w}")
            with st.expander("點擊 + 檢視結果"):
                parse_jisho_senses(w)
                parse_jisho_sentences(w)

if morphology:
    st.markdown("## 詞形變化")
    # Collect inflected forms
    inflected_forms = [tok for tok in doc if tok.tag_.startswith("動詞") or tok.tag_.startswith("形")]
    if inflected_forms:
        create_jap_df(inflected_forms)

if ner_viz:
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")

if tok_table:
    visualize_tokens(doc, attrs=["text", "pos_", "tag_", "dep_", "head"], title="斷詞特徵")
       
