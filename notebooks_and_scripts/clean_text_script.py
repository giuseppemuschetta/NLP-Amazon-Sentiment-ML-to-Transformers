from bs4 import BeautifulSoup
import re
import spacy
import emoji
import nltk
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
import logging

logging.getLogger('nltk').setLevel(logging.ERROR)
from nltk.corpus import stopwords
# per gestire le contrazioni che in inglese sono parecchie e cruciali
import contractions

# Carico il modello spaCy eliminando la componente 'Ner' irrilevante per i nostri scopi.
# Mantenendo solo le funzioni utili come Lemmatizzazione sui token, Pos-tagger e Parser.
# BISOGNA ASSICURARSI DI SCARICARE il modello spaCy usando il seguente comando da terminale:
# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm', disable=['ner'])

# Carico le stop-words in lingua inglese
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# STO TOGLIENDO LE PAROLE FONDAMENTALI PER UNA SENTIMENT ANALYSIS dalle stopwords
important_words = {
    'abysmal', 'acerbic', 'affirmative', 'alarming', 'amazing', 'appalling', 'atrocious', 'awesome', 'awful',
    'bad', 'beautiful', 'beneficial', 'best', 'bitter', 'bleak', 'blissful', 'breathtaking', 'brutal', 'burdensome',
    'captivating', 'celebratory', 'charming', 'cheerful', 'critical', 'daunting', 'deceptive', 'delighted',
    'delightful', 'deplorable', 'depressing', 'detrimental', 'disappointed', 'disastrous', 'disheartening',
    'dismal', 'displeased', 'displeasing', 'disquieting', 'distressing', 'disturbing', 'dolorous', 'dreadful',
    'ecstatic', 'effective', 'elated', 'empowering', 'enchanting', 'enthusiastic', 'euphoric', 'exasperating',
    'excellent', 'exceptional', 'exciting', 'exhilarating', 'exquisite', 'fantastic', 'flawless', 'frightening',
    'frustrating', 'gloomy', 'glorious', 'good', 'gratifying', 'great', 'grim', 'grimace', 'happy', 'harsh', 'hate',
    'heartwarming', 'hopeless', 'horrible', 'horrific', 'important', 'impressed', 'inadequate', 'ineffective',
    'inspiring', 'invigorating', 'joyous', 'jubilant', 'lively', 'love', 'magnificent', 'marvelous', 'mediocre',
    'melancholic', 'menacing', 'miserable', 'morose', 'mournful', 'necessary', 'nurturing', 'oppressive',
    'optimistic', 'outstanding', 'painful', 'pathetic', 'perfect', 'pitiful', 'pleased', 'pleasurable',
    'poor', 'prefer', 'promising', 'radiant', 'recommend', 'refreshing', 'repulsive', 'revitalizing', 'rewarding',
    'ruthless', 'sad', 'satisfied', 'serene', 'shocking', 'sinister', 'sorrowful', 'spectacular', 'splendid',
    'stressing', 'stunning', 'sublime', 'superior', 'terrible', 'terrifying', 'threatening', 'thrilled', 'thriving',
    'tragic', 'tranquil', 'troublesome', 'underwhelming', 'unexpected', 'unfortunately', 'unhappy', 'unsatisfied',
    'uplifting', 'useful', 'useless', 'vexing', 'vibrant', 'woeful', 'wonderful', 'worst', 'worthwhile'
}
stop_words = stop_words - important_words

# AGGIUNGO PAROLE IRRILEVANTI ALLE STOPWORDS
irrelevant_words = {
    'accessory', 'account', 'amazon', 'application', 'around', 'article', 'aspect', 'bar', 'basis', 'batch',
    'bean', 'become', 'beef', 'big', 'black', 'box', 'bowl', 'brew', 'budget', 'button', 'candy', 'capacity', 'cart',
    'case', 'cat', 'category', 'checkout', 'cheese', 'chicken', 'chip', 'chocolate', 'click', 'coconut', 'coffee',
    'color', 'come', 'compare',
    'component', 'consumer', 'content', 'cookie', 'cost', 'couple', 'coverage', 'cup', 'day', 'delivery', 'design',
    'dimension', 'discount', 'dog', 'down', 'drink', 'dry', 'duration', 'entity', 'ever', 'expensive', 'fan', 'feature',
    'food', 'format', 'go', 'grade', 'green', 'guarantee', 'height', 'high', 'house',
    'ingredient', 'inquiry', 'instruction', 'inventory', 'invoice', 'issue', 'item', 'jar', 'kcup', 'keurig', 'length',
    'level', 'local', 'location', 'make', 'manual', 'manufacturer', 'market', 'mass', 'material', 'merchant', 'minute',
    'mode', 'model', 'month', 'natural', 'offer', 'oil', 'one', 'online', 'only', 'option', 'organic', 'outlet', 'over',
    'oz', 'pack', 'package', 'packet', 'payment', 'peanut', 'percent', 'place', 'policy', 'pop', 'position', 'pound',
    'powder', 'pretty', 'provider', 'put', 'quantity', 'quality', 'range', 'rate', 'regular', 'requirement', 'retailer',
    'return', 'roast', 'salt', 'save', 'scale', 'scope', 'search', 'seem', 'selection', 'series',
    'setting', 'shipping', 'shop', 'since', 'site', 'size', 'small', 'solution', 'some', 'soup', 'stage', 'standard',
    'start', 'status', 'stock', 'store', 'structure', 'stuff', 'style', 'sugar', 'sure', 'surface', 'system', 'take',
    'tea', 'technology', 'texture', 'though', 'time', 'too', 'top', 'transaction', 'trend', 'tutorial', 'type', 'unit',
    'use', 'user', 'utility', 'value', 'vanilla', 'variety', 'vendor', 'version', 'very', 'volume', 'warranty', 'way',
    'website', 'while', 'whole', 'width', 'wish', 'year', 'yet', 'zone'
}
stop_words = stop_words.union(irrelevant_words)


# GESTIONE DELLE EMOJI
def is_emoji(s):
    return any(c in emoji.EMOJI_DATA for c in s)


#end_function


# DEFINZIONE DELLA FUNZIONE TOTALE DI CLEANING:
def clean_text(text):
    """
    - Converts all text to lowercase to standardize case sensitivity.
    - Retains emojis detected in the text as they can enhance sentiment analysis.
    - Uses BeautifulSoup to remove HTML tags, which are not useful for text analysis.
    - Uses regular expressions to remove URLs and email addresses, as these are non-relevant details for most text analysis tasks.
    - Expands contractions (e.g., "don't" to "do not") to facilitate analysis and comparison of the text.
    - Removes numbers, special characters, and punctuation to reduce noise in the text and focus on verbal content.
    - Deletes mentions and hashtags if the text originates from social media, as these can distort the analysis based on common linguistic content.
    - Compresses multiple spaces into a single space and removes leading and trailing spaces to standardize formatting.
    - Uses spaCy to tokenize the text, apply lemmatization, and filter tokens based on part of speech and presence in stop-word lists, retaining only relevant parts of speech such as nouns, verbs, adjectives, and adverbs.
    - Applies an additional filter to remove tokens that are punctuation, spaces, or numbers, and not present in stop-word lists.
    - Handles negatives!
    - Identifies and combines significant bigrams using Gensim, which can reveal meaningful word combinations for analysis.

    @Args:
        text (str): A string containing the text to be cleaned.

    @Returns:
        str: The cleaned text, still in string form, containing lemmatized words separated by spaces, enriched with relevant bigrams.
"""
    # CONTROLLO DI EMPTYNESS:
    if not text:
        return ""

    # RIMOZIONE TAG HTML
    # Utilizza BeautifulSoup per analizzare il testo dato e rimuovere eventuali tag HTML/XML.
    text = BeautifulSoup(text, "html.parser").get_text()

    # CONVERSIONE DEL TESTO IN MINUSCOLO
    text = text.lower()

    # ESPANSIONE DELLE CONTRAZIONI
    text = contractions.fix(text)

    # PULIZIA ULTERIORE
    # Rimozione di indirizzi email e URL dal testo
    text = re.sub(r'\b\w+@\w+\.\w+\b|http[s]?://\S+', '', text)

    # Rimozione di numeri, caratteri non alfabetici dal testo
    text = re.sub(r'\b\d+\b|[^\w\s]|\'s\b', '', text)

    # Rimozione di spazi multipli e spazi all'inizio e alla fine del testo,
    # assicurandosi però che vi sia solo uno spazio tra le parole
    text = re.sub(r'\s+', ' ', text).strip()

    # ELABORAZIONE DEL TESTO CON SPACY
    doc = nlp(text)
    filtered_sentence = []

    # GESTIONE NEGAZIONI, CONSERVAZIONE DELLE EMOJI E FILTRAGGIO DEI TOKEN VALIDI
    negation_tokens = {"not", "never", "no"}
    exceptions_token_text = {'not_have', 'not_as'}
    i = 0
    while i < len(doc):
        # Estraggo il token corrente
        token = doc[i]
        # Se il token corrente è una emoji, la mantengo
        if is_emoji(token.text):
            filtered_sentence.append(token.text)
            i += 1
        # Se il token corrente è un 'no' o 'never' o 'not' allora è idoneo ad essere gestito.
        # Il token verrà unito al successivo mediante un underscore: ad es not_bad, no_talking ecc...
        elif token.lower_ in negation_tokens and i + 1 < len(doc):
            next_token = doc[i + 1]
            token_text = f"{token.text}_{next_token.text}"
            if token_text not in exceptions_token_text:
                filtered_sentence.append(token_text)
            i += 2
        else:
            # Qui avviene il filtraggio vero e proprio basato sulle POS, stop words, punteggiatura ecc...
            # Se il token corrente è un aggettivo, avverbio, sostantivo o verbo e non appartiene
            # alla lista delle stopwords e non è nemmeno punteggiatura, spazio e digit, allora lo manteniamo!
            if (token.pos_ in ['ADJ', 'ADV', 'NOUN', 'VERB']  # ['ADJ', 'ADV', 'NOUN', 'VERB']
                    and token.lemma_ not in stop_words
                    and not token.is_punct
                    and not token.is_space
                    and not token.is_digit):
                filtered_sentence.append(token.lemma_)
            i += 1
    #end_while

    # RILEVAZIONE COLLOCATIONS CON GENSIM (Bigrams detection)
    phrases = Phrases([filtered_sentence],
                      scoring='default',
                      min_count=1,
                      threshold=10,
                      connector_words=ENGLISH_CONNECTOR_WORDS)
    bigram = Phraser(phrases)
    bigrams = bigram[filtered_sentence]

    return " ".join(bigrams)
#end_clean_text_function


# Example usage
#review = "Here's an example: I'm not just excited, but super excited!!! 😊 Visit us at http://example.com or call the number 123-456-789"
#cleaned_text = clean_text(review)
#print(cleaned_text)
#%%
