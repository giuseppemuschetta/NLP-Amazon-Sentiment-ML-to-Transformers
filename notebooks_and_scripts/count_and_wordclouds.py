import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

def generate_wordcloud(df, column_name, sentiment, color, wordcloud_name):
    """
    Genera e visualizza un WordCloud per le recensioni basate sul loro sentiment.

    @args:
    df (pandas.DataFrame): DataFrame contenente le recensioni e i loro sentiment.
    column_name (str): Nome della colonna contenente il testo, che può essere raw o ripulito, delle recensioni.
    sentiment (int): Il valore del sentiment delle recensioni da visualizzare (e.g., 1 per positivo, 0 per negativo).
    color (str): Colore di sfondo del WordCloud.
    wordcloud_name (str): Nominativo della wordcloud specifica.

    @returns:
    None: La funzione visualizza un WordCloud e non ritorna nessun valore.
    """
    # tramite il nominativo fornito in input genero path diversi
    actual_path = "Images/Preparation/" + wordcloud_name + ".png"

    review = ' '.join(df[df['sentiment'] == sentiment][column_name])

    wc = WordCloud(background_color=color, width=800, height=400)

    plt.figure(figsize=(10, 7))

    plt.imshow(wc.generate(review), interpolation='bilinear')

    if sentiment == 1:
        plt.title('Parole più comuni in recensioni positive')
    else:
        plt.title('Parole più comuni in recensioni negative')

    plt.axis('off')
    plt.savefig(actual_path, dpi=300, bbox_inches='tight')
    plt.show()
#end_generate_wordcloud_function


def count_most_common_words(df, column_name, sentiment, max_words=50):
    """
    Conta e stampa le parole più frequenti per un dato sentiment da un DataFrame.

    @args:
    df (pandas.DataFrame): DataFrame contenente le recensioni e i loro sentiment.
    column_name (str): Nome della colonna contenente il testo, che può essere raw o rupulito, delle recensioni.
    sentiment (int): Il valore del sentiment per cui contare le parole (1 per positivo, 0 per negativo).
    max_words (int): Numero massimo di parole da visualizzare.

    @returns:
    list: Ritorna una lista delle tuple contenenti le parole e le loro frequenze.
    """
    if sentiment == 1:
        reviews = df[df['sentiment'] == 1][column_name]
    else:
        reviews = df[df['sentiment'] == 0][column_name]

    # Tokenizza le recensioni
    words = reviews.apply(word_tokenize)

    # Conteggio delle parole
    word_counts = Counter()
    for word_list in words:
        word_counts.update(word_list)

    # Mostra le max_words parole più frequenti
    most_common_words = word_counts.most_common(max_words)

    print(f"Le {max_words} parole più frequenti nelle recensioni positive sono:")
    for word, freq in most_common_words:
        print(f"{word}: {freq}")
    return most_common_words
#end_count_most_common_words_function
#%%

#%%
