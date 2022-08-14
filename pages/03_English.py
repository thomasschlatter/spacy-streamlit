import pandas as pd
import re
import requests 
import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
#from spacy.language import Language
from spacy.tokens import Doc
import spacy_ke
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 
import streamlit as st
import nltk
from nltk.corpus import wordnet as wn

# Download NLTK data 
nltk.download('wordnet')
nltk.download('omw') # standing for Open Multilingual WordNet

# Global variables
EN_TEXT = """(Reuters) Taiwan's government believes there is "enormous" room for cooperation with the European Union on semiconductors, responding to plans from the bloc to boost its chip industry and cut its dependence on U.S. and Asian supplies.
The EU's plan mentions Taiwan, home to the world's largest contract chipmaker TSMC and other leading semiconductor companies, as one of the "like-minded partners" Europe would like to work with.
The plan, unveiled on Tuesday, calls for the European Commission to ease funding rules for innovative semiconductor plants, a move that comes as a global chip shortage and supply chain bottlenecks have created havoc for many industries.
Taiwan's Foreign Ministry said in a statement it was pleased to see the strong momentum in bilateral trade and investment between Taiwan and the EU, and welcomed the EU attaching so much importance to the island."""
DESCRIPTION = "AI模型輔助語言學習"
LOADED_MODEL = "en_core_web_sm"
TOK_SEP = " | "
    
# Utility functions
def create_eng_df(tokens):
    seen_texts = []
    filtered_tokens = []
    for tok in tokens:
        if tok.lemma_ not in seen_texts:
            filtered_tokens.append(tok)
            
    df = pd.DataFrame(
      {
          "單詞": [tok.text.lower() for tok in filtered_tokens],
          "詞類": [tok.pos_ for tok in filtered_tokens],
          "原形": [tok.lemma_ for tok in filtered_tokens],
      }
    )
    st.dataframe(df)
    csv = df.to_csv().encode('utf-8')
    st.download_button(
      label="下載表格",
      data=csv,
      file_name='eng_forms.csv',
      )
          
def filter_tokens(doc):
    clean_tokens = [tok for tok in doc if tok.pos_ not in ["PUNCT", "SYM"]]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_email]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_url]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_num]
    clean_tokens = [tok for tok in clean_tokens if not tok.is_punct]
    clean_tokens = [tok for tok in clean_tokens if not tok.is_space]
    return clean_tokens

def get_def_and_ex_from_wordnet(word, pos_label):
    mapper = {
        "VERB": wn.VERB,
        "NOUN": wn.NOUN,
        "ADJ": wn.ADJ,
        "ADV": wn.ADV,
    }
    
    synsets = wn.synsets(word, pos=mapper[pos_label])
    if len(synsets) > 3:
        synsets = synsets[:3]
    
    sense_count = 1
    for syn in synsets:
        st.write(f"Relevant Sense {sense_count}: {syn.definition()}")
        sense_count += 1
        examples = syn.examples()
        if not examples:
            continue
        elif len(examples) > 3:
            examples = examples[:3]
        ex_count = 1
        for ex in examples:
            st.write(f"Relevant Example {ex_count} >>> {ex}")
            ex_count += 1    
        st.markdown("---")
  
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

# Load the language model
nlp = spacy.load(LOADED_MODEL)

# Add pipelines to spaCy
nlp.add_pipe("yake") # keyword extraction
nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang}) # WordNet
# nlp.add_pipe("merge_entities") # Merge entity spans to tokens

# Page starts from here
st.markdown("## 待分析文本")     
st.info("請在下面的文字框輸入文本並按下Ctrl + Enter以更新分析結果")
text = st.text_area("",  EN_TEXT, height=200)
doc = nlp(text)
st.markdown("---")

# Checkboxes for various features
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
        enriched_sentence = []
        for tok in sent:
            if tok.pos_ != "VERB":
                enriched_sentence.append(tok.text)
            else:
                synsets = tok._.wordnet.synsets()
                if synsets:
                    v_synsets = [s for s in synsets if s.pos()=='v']
                    if not v_synsets:
                        enriched_sentence.append(tok.text)
                    else:
                        lemmas_for_synset = [lemma for s in v_synsets for lemma in s.lemma_names()]
                        lemmas_for_synset = list(set(lemmas_for_synset))

                        try:
                            lemmas_for_synset.remove(tok.text)
                        except:
                            pass

                        if len(lemmas_for_synset) > 5:
                            lemmas_for_synset = lemmas_for_synset[:5]

                        lemmas_for_synset = [s.replace("_", " ") for s in lemmas_for_synset]
                        lemmas_for_synset = " | ".join(lemmas_for_synset)
                        enriched_tok = f"{tok.text} (cf. {lemmas_for_synset})"
                        enriched_sentence.append(enriched_tok)  


        display_text = " ".join(enriched_sentence)
        st.write(f"{idx+1} >>> {display_text}")     

if defs_examples:
    st.markdown("## 單詞解釋與例句")
    clean_tokens = filter_tokens(doc)
    num_pattern = re.compile(r"[0-9]")
    clean_tokens = [tok for tok in clean_tokens if not num_pattern.search(tok.lemma_)]
    selected_pos = ["VERB", "NOUN", "ADJ", "ADV"]
    clean_tokens = [tok for tok in clean_tokens if tok.pos_ in selected_pos]
    tokens_lemma_pos = [tok.lemma_ + " | " + tok.pos_ for tok in clean_tokens]
    vocab = list(set(tokens_lemma_pos))
    if vocab:
        selected_words = st.multiselect("請選擇要查詢的單詞: ", vocab, vocab[0:3])
        for w in selected_words:
            word_pos = w.split("|")
            word = word_pos[0].strip()
            pos = word_pos[1].strip()
            st.write(f"### {w}")
            with st.expander("點擊 + 檢視結果"):
                get_def_and_ex_from_wordnet(word, pos)

if morphology:
    st.markdown("## 詞形變化")
    # Collect inflected forms
    inflected_forms = [tok for tok in doc if tok.text.lower() != tok.lemma_.lower()]
    if inflected_forms:
        create_eng_df(inflected_forms)

if ner_viz:
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")

if tok_table:
    visualize_tokens(doc, attrs=["text", "pos_", "tag_", "dep_", "head"], title="斷詞特徵")
