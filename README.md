# NBmoodclassification
Sentiment Analysis on Songs based on Song Lyrics using Naïve Bayes Algorithm

Music induces basic to complex emotions such as happiness, sadness, and nostalgia. These emotions can be classified into categories like positive or negative using sentiment analysis. Existing studies on mood classification mostly focus on the audio features of a song while the lyric features are ignored. A few studies on lyrics mood classification, on the other hand, pointed out the need to explore other classifiers like Naïve Bayes and improve its performance, using a larger dataset. In this study, a Naïve Bayes classifier model was created to identify whether a song is positive or negative based on its lyrics. The model which produced exceptional results with 95.02% accuracy and 94.42% precision was trained and tested using a dataset containing 1,810 song lyrics. Feature extraction techniques such as N-grams (tri-grams) and TF-IDF were applied after preprocessing the data.

## NOTES ON EACH MODULE
### lyricscraper.py
This module uses the _lyricsgenius_ library to search for a song lyric from the Genius.com website. After searching, checking is done to identify if the lyrics are in English. After all song lyrics are searched, it is saved in a CSV file as a data frame.
    DEPENDENCIES:
      -  Run the following command(s):
          `pip install lyricsgenius`
          `pip install langdetect`
          `pip install pandas`
### preprocessing.py
This module uses NLTK to clean and preprocess the lyrics.
    DEPENDENCIES:
      - Run the following command(s):
          `pip install nltk`
      - Uncomment the commented section to download the needed NLTK corpora.
### naivebayes.py
This module trains and tests the preprocessed data using the Multinomial Naive Bayes classifier using _scikit-learn_.
    DEPENDENCIES:
      - Run the following command(s):
          `pip install scikit-learn`
### main.py
This module reads the CSV file, cleans and preprocesses the data by calling the preprocessing module, trains the data by calling the naivebayes module, and tests a song using PySimpleGUI.
    DEPENDENCIES:
      - Run the following command(s):
          `pip install seaborn`
          `pip install wordcloud`
          `pip install PySimpleGUI`
  
