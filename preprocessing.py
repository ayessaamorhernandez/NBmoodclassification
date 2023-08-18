import nltk #installed for preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
 
import main
import re

# NOTE: UNCOMMENT THIS SECTION IF BEING RUN FOR THE FIRST TIME
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

stop_words = stopwords.words('english') + ['yeah', 'ohh', 'ahh', 'aah', 'ooh', 'oooh']
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean(lyrics):
    cleaned = []    
    for lyric in lyrics:
        lowered = lyric.lower() #lowercase the lyrics
        not_lyrics = re.sub("contributor.*lyrics|embed|intro|chorus|repeat|verse|bridge|pre-chorus","", lowered)
        numbers_removal = re.sub("[0-9]", "", not_lyrics)
        in_brackets = re.sub("\[.*\]|[\(\)!?\.\,-]"," ", numbers_removal) #removing words inside the square brackets and special characters
        quotes = in_brackets.replace("'", " ")
        back_tick = quotes.replace("`", " ")
        cleaned.append(back_tick)
    return cleaned

def preprocess(cleaned_lyrics):
    tokenized = []
    for c in cleaned_lyrics:
        tokenize = word_tokenize(c)
        tokenized.append(tokenize)

    stopwords_removed = []
    for t in tokenized:
        sw = []
        for w in t:
            if w not in stop_words and len(w)>2:
                sw.append(w)    
            
        stopwords_removed.append(sw)

    # stemmed = []
    # for sr in stopwords_removed:
    #     s = []
    #     for w in sr:
    #         s.append(stemmer.stem(w))
    #     stemmed.append(s)

    lemmatized = []
    for sr in stopwords_removed:
        s = []
        for w in sr:
            s.append(lemmatizer.lemmatize(w))
        lemmatized.append(s)

    joined = []
    for l in lemmatized:
        j = " ".join(l)
        joined.append(j)
    
    return joined