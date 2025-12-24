import string
import re
import emoji
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from wordcloud import STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from joblib import dump
import nltk
def cleaning_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # menghapus mention
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # menghapus hashtag
    text = re.sub(r'RT[\s]', '', text) # menghapus RT
    text = re.sub(r"http\S+", '', text) # menghapus link
    text = re.sub(r'[0-9]+', '', text) # menghapus angka
    text = re.sub(r'[^\w\s]', '', text) # menghapus karakter selain huruf dan angka
 
    text = text.replace('\n', ' ') # mengganti baris baru dengan spasi
    text = text.translate(str.maketrans('', '', string.punctuation)) # menghapus semua tanda baca
    text = text.strip(' ') # menghapus karakter spasi dari kiri dan kanan teks
    return text

def casefolding_text(text): # Mengubah semua karakter dalam teks menjadi huruf kecil
    text = text.lower()
    return text

def tokenizing_text(text): # Memecah atau membagi string, teks menjadi daftar token
    text = word_tokenize(text)
    return text

def filtering_text(text):
    list_stop_words_eng = set(STOPWORDS)
   
    filtered = []
    for txt in text:
        if txt not in list_stop_words_eng:
            filtered.append(txt)
    
    return filtered

def to_sentence(list_words):
    sentence = ' '.join(word for word in list_words)
    return sentence

def remove_emojis(text):
    return ''.join(char for char in text if char not in emoji.EMOJI_DATA)

def data_preprocessing(df_path, target_cols, save_path, test_size=0.2):
    df = pd.read_csv(df_path)
    df = df.dropna()
    df = df.drop_duplicates()
    df['clean_text'] = df['Tweet'].apply(remove_emojis)\
                              .apply(cleaning_text)\
                              .apply(casefolding_text)\
                              .apply(tokenizing_text)\
                              .apply(filtering_text)\
                              .apply(to_sentence)
    
    df['clean_text'] = df['clean_text'].astype(str)
    text_transform = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), lowercase=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transform, 'clean_text'),
        ],
        remainder='drop'
    )

    X = df[['clean_text']]
    label_encod = LabelEncoder()
    y = label_encod.fit_transform(df[target_cols])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    dump(preprocessor, save_path)
    dump(X_train, "../data/X_train.pkl")
    dump(X_test,  "../data/X_test.pkl")
    dump(y_train, "../data/y_train.pkl")
    dump(y_test,  "../data/y_test.pkl")
    return X_train, X_test, y_train, y_test, df

def main():
    CSV_PATH = 'hf://datasets/nikesh66/Sarcasm-dataset/sarcasm_tweets.csv'
    nltk.download("punkt")
    nltk.download("punkt_tab") 
    data_preprocessing(
        df_path=CSV_PATH, 
        target_cols='Sarcasm (yes/no)', 
        save_path='artifacts/preprocessing.joblib', 
        test_size=0.2
    )
if __name__ == '__main__':
    main()
