import re
import string

#I: CLEANING

def remove_url(text):
  '''
  remove urls from a string
  '''
  return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_html(text):
  '''
  remove htmls from a string
  '''
  html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
  return re.sub(html, "", text)

def remove_non_ascii(text):
  '''
  remove non ASCII characters from a string
  '''
  return re.sub(r'[^\x00-\x7f]',r'', text)

def remove_special_characters(text):
  """
  remove special characters from a string
  """
  emoji_pattern = re.compile(
    '['
    u'\U0001F600-\U0001F64F'  # emoticons
    u'\U0001F300-\U0001F5FF'  # symbols & pictographs
    u'\U0001F680-\U0001F6FF'  # transport & map symbols
    u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
    u'\U00002702-\U000027B0'
    u'\U000024C2-\U0001F251'
    ']+',
    flags=re.UNICODE)
    
  return emoji_pattern.sub(r'', text)

def remove_punctuations(text):
  '''
  Params:
  text: a string
  Returns:
  the same string with punctuations removed
  '''
  return text.translate(str.maketrans('', '', string.punctuation))

def clean(text):
  '''
  completely clean a string
  '''
  text = remove_url(text)
  text = remove_html(text)
  text = remove_special_characters(text)
  text = remove_non_ascii(text)
  text = remove_punctuations(text)

  return text

#II: TOKENIZATION

#1) HuggingFace
import transformer
from transformer import AutoTokenizer

def bert_tokenize(text):
  '''
  tokenized a text using BERT auto tokenizer
  '''
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

  return tokenizer.bert_tokenize(text)

#2) Natural Language Toolkit
from nltk.tokenize import word_tokenize

def nltk_tokenize(text):
  '''
  tokenize a text using nltk tokenizer
  '''
  return word_tokenize(text)

#3) Textblob
from textblob import TextBlob

def textblob_tokenize(text):
  '''
  tokenize using textblob tokenizer
  '''
  blob = TextBlob(text)

  return blob.words

#4) Gensim
from gensim.utils import simple_preprocess

def gensim_tokenize(text):
  '''
  tokenize using gensim tokenizer
  '''
  return simple_preprocess(text)

#5) Keras
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])

def keras_tokenizer(text):
  '''
  tokenize using keras tokenizer
  '''
  return tokenizer.texts_to_sequences([text])

#6) SentencePiece
import sentencepiece as spm

spm.SentencePieceTrainer.train('--input=your_text_file.txt --model_prefix=m --vocab_size=5000')
sp = spm.SentencePieceProcessor(model_file='m.model')

def sentencepiece_tokenize(text):
  '''
  tokenize using SentencePiece
  '''
  return sp.encode(text, out_type=str)

#7) spaCy
import spacy

nlp = spacy.load("en_core_web_sm")

def spacy_tokenize(text):
  '''
  tokenize using spaCy
  '''
  doc = nlp(text)

  return [token.text for token in doc]

#III: REMOVE STOP-WORDS

#1) Natural Language Toolkit
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokenized_text):
  '''
  removing stop-words using nltk
  '''
  return [word for word in tokenized_text if word not in stop_words]

#2) spaCy
import spacy

nlp = spacy.load("en_core_web_sm")

def remove_stopwords_spacy(tokenized_text):
  '''
  removing stop-words using spacy
  '''
  return [word for word in tokenized_text if not nlp.vocab[word].is_stop]

#3) scikit-learn
from sklearn.feature_extraction.text import CountVectorizer

def remove_stopwords_sklearn(text):
  '''
  removing stop-words using sklearn
  '''
  vectorizer = CountVectorizer(stop_words='english')

  return vectorizer.build_analyzer()(text)

#4) Gensim
from gensim.parsing.preprocessing import remove_stopwords

def remove_stopwords_gensim(text):
  '''
  removing stop-words using gensim
  '''
  return remove_stopwords(text)

#5) Customization
custom_stop_words = {'a', 'an', 'the'} #specifying own's list

def remove_stopwords_custom(tokenized_text):
  '''
  removing stop-words using own's list
  '''
  return [word for word in tokenized_text if word not in custom_stop_words]

#IV) STEMMING: Natural Language Toolkit

from nltk.stem import PorterStemmer

stemmer = nltk.PorterStemmer()

def porter_stemmer(text):
  """
  stem a text using porter stemmer
  """
  stems = [stemmer.stem(i) for i in text]

  return stems

from nltk.stem import SnowballStemmer

stemmer = nltk.SnowballStemmer()

def snowball_stemmer(text):
  """
  stem a text using porter stemmer
  """
  stems = [stemmer.stem(i) for i in text]

  return stems

from nltk.stem import LancasterStemmer

stemmer = nltk.LancasterStemmer()

def lancaster_stemmer(text):
  """
  stem a text using porter stemmer
  """
  stems = [stemmer.stem(i) for i in text]

  return stems

#V: POS-TAGGING: Natural Language Toolkit

from nltk.corpus import wordnet
nltk.download('wordnet')
from nltk.corpus import brown
nltk.download('brown')

wordnet_map = {
  "N":wordnet.NOUN,
  "V":wordnet.VERB,
  "J":wordnet.ADJ,
  "R":wordnet.ADV
}

train_sents = brown.tagged_sents(categories='news')
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)

def pos_tag_wordnet(text, pos_tag_type="pos_tag"):
  """
  create pos_tag with wordnet format
  """
  pos_tagged_text = t2.tag(text)
  pos_tagged_text = [(word, wordnet_map.get(pos_tag[0])) if pos_tag[0] in wordnet_map.keys() else (word, wordnet.NOUN) for (word, pos_tag) in pos_tagged_text]

  return pos_tagged_text

#VI: LEMMATIZATION

#1) Natural Language Toolkit
from nltk.stem import WordNetLemmatizer

def nltk_lemmatize(text):
  """
  lemmatize the tokenized words using 
  """

  lemmatizer = WordNetLemmatizer()
  lemma = [lemmatizer.lemmatize(word, tag) for word, tag in text]
  
  return lemma

#with POS-tagging
train_df['lemmatize_word_w_pos'] = train_df['combined_postag_wnet'].apply(lambda x: lemmatize_word(x))
train_df['lemmatize_text'] = [' '.join(map(str, l)) for l in train_df['lemmatize_word_w_pos']]

#2) TextBlob
from textblob import TextBlob

def textblob_lemmatize(text):
  '''
  lemmatize text using textblob
  '''
  blob = TextBlob(text)

  return [word.lemmatize() for word in blob.words]

#3) spaCy
import spacy

nlp = spacy.load("en_core_web_sm")

def spacy_lemmatize(text):
  '''
  lemmatize text using spacy
  '''
  doc = nlp(text)

  return [token.lemma_ for token in doc]

#VII: VECTORIZATION

#1) Scikit-Learn
corpus = train_df["lemmatize_text"].tolist()

#count vectorization
from sklearn.feature_extraction.text import CountVectorizer

def cv(data, ngram, MAX_NB_WORDS):
  count_x = CountVectorizer(ngram_range = (ngram, ngram), max_features = MAX_NB_WORDS)
  emb = count_x.fit_transform(data).toarray()

  return emb, count_x

#tfidf vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

def TFIDF(data, ngram, MAX_NB_WORDS):
  tfidf_x = TfidfVectorizer(ngram_range = (ngram, ngram), max_features = MAX_NB_WORDS)
  emb = tfidf_x.fit_transform(data).toarray()

  return emb, tfidf_x

#2) gensim

import gensim
import numpy as np

def get_average_vec(tokens_list, vector, generate_missing=False, k=300):
  """
  Calculate average embedding value of sentence from each word vector
  """  
  if len(tokens_list)<1:
    return np.zeros(k)
    
  if generate_missing:
    vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
  else:
    vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    
  length = len(vectorized)
  summed = np.sum(vectorized, axis=0)
  averaged = np.divide(summed, length)

  return averaged

def get_embeddings(vectors, text, generate_missing=False, k=300):
  """
  create the sentence embedding
  """
  embeddings = text.apply(lambda x: get_average_vec(x, vectors, generate_missing=generate_missing, k=k))  
  
  return list(embeddings)

#word2vec

word2vec_path = "../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin"

# we only load 200k most common words from Google News corpus 
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=200000)

embeddings_word2vec = get_embeddings(word2vec_model, train_df["lemmatize_text"], k=300)

#gloVe

from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = "../input/glove6b/glove.6B.300d.txt"
word2vec_output_file = "glove.6B.100d.txt.word2vec"
glove2word2vec(glove_input_file, word2vec_output_file)

# we only load 200k most common words from Google New corpus 
glove_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False, limit=200000) 

embeddings_glove = get_embeddings(glove_model, train_df["lemmatize_text"], k=300)

#3) HuggingFace: advanced embeddings method

import tensorflow_hub as hub

# download the tonkenizer 
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tokenization

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)

def bert_encode(texts, tokenizer, max_len=512):
  all_tokens = []
  all_masks = []
  all_segments = []
    
  for text in texts:
    text = tokenizer.tokenize(text)      
    text = text[:max_len-2]
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    pad_len = max_len - len(input_sequence)
        
    tokens = tokenizer.convert_tokens_to_ids(input_sequence)
    tokens += [0] * pad_len
    pad_masks = [1] * len(input_sequence) + [0] * pad_len
    segment_ids = [0] * max_len
    
    all_tokens.append(tokens)
    all_masks.append(pad_masks)
    all_segments.append(segment_ids)
    
  return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

bert_input = bert_encode(train_df["text"].values, tokenizer, max_len=300)

#VIII: PADDING

#1) Keras
from keras.preprocessing.sequence import pad_sequences

padded_emb = pad_sequences(vectorized_emb, maxlen = max_len)

#3) HuggingFace
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


encoding = tokenizer(texts, padding=True, truncation=True, max_length=10, return_tensors='pt')
padded_sequences = encoding['input_ids']
