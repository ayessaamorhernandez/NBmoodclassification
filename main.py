import seaborn as sns #installed for data visualization
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud #installed for generating wordcloud/data visualization
import PySimpleGUI as sg #installed for gui

import lyricscraper as ls
import preprocessing as pp
import naivebayes as nb

TITLE = 'title'
ARTIST = 'artist'
MOOD = 'mood'
LYRICS = 'lyrics'
PREPROCESSED = 'preprocessed lyrics'
CLEANED = 'cleaned lyrics'

sg.theme('LightBrown6')

def initial_visualization(df):
    sns.catplot(x=MOOD, data=df, kind = 'count')
    plt.show()

def wordCloud(df):
    pos_text = neg_text = ""
    for x in df.values:
        if x[2] == 'pos':
            pos_text += x[4]
        if x[2] == 'neg':
            neg_text += x[4]
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(pos_text)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(neg_text)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def classify(vect, model):
    layout = [ [sg.Text('Enter song title: '), sg.InputText(key=TITLE)],
            [sg.Text('Enter singer/artist: '), sg.InputText(key=ARTIST)],
            [sg.Text('Classification: '), sg.Text(key=MOOD, font=("Georgia"))],
            [sg.Text(key='-FOUND-')],
            [sg.Multiline(size=(100, 30), key=LYRICS, disabled=True)],
            [sg.Button('Classify'), sg.Button('Refresh'), sg.Button('Exit')] ]
    
    win = sg.Window('Sentiment Analysis on Songs based on Song Lyrics using NB Classifier', layout)

    while True:
        event, values = win.read()
        if event == 'Classify':
            test_title = values[TITLE]
            test_artist = values[ARTIST]
            song = ls.lyrics_genius.search_song(test_title, test_artist)
            if song:
                win['-FOUND-'].update('Lyrics found!')
                cleaned = pp.clean([song.lyrics])
                preprocessed = pp.preprocess(cleaned)
                input_x = vect.transform(preprocessed)
                pred = model.predict(input_x)
                p = "POSITIVE" if pred[0] == 1 else "NEGATIVE"
                print(f"============== CLASSIFICATION ============== \nThe song is classified as: {p} \n")
                print(f"=============== LYRICS FOUND =============== \n{song.lyrics}\n")
                win[MOOD].update(p)
                win[LYRICS].update(song.lyrics)
            else:
                win['-FOUND-'].update('Lyrics NOT found! Try again.')
                print("Song lyrics not found. Try again.\n")
        if event == 'Refresh':
            win[TITLE].update('')
            win[ARTIST].update('')
            win[MOOD].update('')
            win[LYRICS].update('')
            win['-FOUND-'].update('')
        if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
            break
    win.close()


def main():
    print("Reading the dataset.....")
    df = pd.read_csv("with_lyrics.csv", index_col='index')
    
    # initial_visualization(df)
    print("Preprocessing the data.....")
    cleaned = pp.clean(df[LYRICS])
    preprocessed = pp.preprocess(cleaned)
    
    df[PREPROCESSED] = preprocessed
    df[CLEANED] = cleaned

    # wordCloud(df)

    print("Training and testing the classifier.....")
    vect, model = nb.train(df)
    

    # while True:
    classify(vect, model)

if __name__ == '__main__':
    main()