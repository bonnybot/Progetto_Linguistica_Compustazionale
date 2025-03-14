import sys
import nltk
from nltk.stem import WordNetLemmatizer
import pickle

def read_file_contents(file_path): #apro il file in lettura con codifica utf8 
    with open(file_path, "r", encoding="utf8") as infile:
        contents = infile.read()
    return contents

def sentence_splitting(text): #suddivido il testo in frasi
    sentences = nltk.tokenize.sent_tokenize(text)
    return sentences

def get_tokens(text): #suddivido le frasi del testo in token
    sentences = nltk.tokenize.sent_tokenize(text)
    all_tokens = []
    for sentence in sentences:
        sentence_tokens = nltk.tokenize.word_tokenize(sentence)
        all_tokens = all_tokens + sentence_tokens
    return all_tokens #lista con tutti i token

def stemming(tokens): #lemmatizzazione dei token
    stemmer = WordNetLemmatizer()
    stems = [stemmer.lemmatize(token) for token in tokens] #list comprehension per lemmatizzare ogni singolo token

    return stems

def annotate(sentences): #annotazione linguistica del testo (sentence splitting, token PoS taggati, lemmatizzazione)
    for sentence in sentences:
        tokens = nltk.tokenize.word_tokenize(sentence)
        tokens_PoS = nltk.tag.pos_tag(tokens)
        print(f"La frase è: \033[3m {sentence} \033[0m") #inizio e fine corsivo
        print(f"I token PoS taggati che la compongono sono: {tokens_PoS}")
        print(f"I lemmi dei token nella frase sono: {stemming(tokens)}\n")

def average_sentences_length(sentences, tokens): 
    length = len(tokens)
    average_length = length*1.0/len(sentences)*1.0 #calcolo lunghezza media delle frasi in token

    return average_length

def average_tokens_length(tokens): #calcolo lunghezza media dei token
    lenght = 0
    characters_length = 0
    for token in tokens:
        if token not in [",",".",";","!",":","'","?"]: #escludo la punteggiatura
            tokens_length = len(token)
            characters_length += tokens_length #sommo la lunghezza di ogni token a quella dei caratteri 
            lenght+=1
    average = float(characters_length)/lenght*1.0

    return average

def get_hapax(tokens): #calcolo degli hapax
    hapax = []
    for token in list(set(tokens)): #con set escludo i duplicati 
        token_freq = tokens.count(token)
        if (token_freq == 1) and (len(token)>1): #considero solo i token presenti una volta e più lunghi di un carattere (per escludere in modo grossolano la punteggiatura)
            hapax+= token

    return hapax

def count_hapax(vocabulary): #conto gli hapax
    count = 0
    for hapax in vocabulary:
        count+=1

    return count

def analyze_slice(tokens, upper_bound): #calcolo la distribuzione degli hapax in uno specifico intervallo
    tokens_slice = tokens[:upper_bound]
    hapax = get_hapax(tokens_slice)
    distribution_hapax = len(hapax)/len(tokens_slice) #lunghezza degli hapax fratto la lunghezza dei token nell'intervallo

    return distribution_hapax

def get_TTR_slice(tokens, upper_bound = 200): #calcolo la ricchezza lessicale in uno specifico intervallo
    tokens_slice = tokens[:upper_bound]
    vocabulary = list(set(tokens_slice))
    TTR = len(vocabulary)/len(tokens_slice) #lunghezza del vocabolario fratto la lunghezza dei token nell'intervallo
    print(f"La Type-Token Ratio per i primi {upper_bound} token è: {TTR:.2f}") #specifico l'approssimazione del float

def pos_tagging(tokens): #PoS tagging dei tokens
    token_PoS = nltk.tag.pos_tag(tokens) #di default language è settato sull'inglese

    return token_PoS

def classifier(sentencesFile): #classificatore di polarità 
    sentiment_pipeline = pickle.load(open("sentiment-classifier.pkl", "rb")) #carico con pickle il classificatore
    POS = []
    NEG = []
    countNEG = 0
    countPOS = 0
    for sentence in sentencesFile: #itero sulle frasi del file
        pred = sentiment_pipeline.predict([sentence]) #utilizzo il classificatore per predire nuove frasi
        if pred == [0]: #nel classificatore il valore 0 corrisponde alle frasi con polarità negativa
            countNEG +=1
            NEG.append(sentence) #conto le frasi con polarità negativa e poi le inserisco nella lista
        elif pred == [1]: #nel classificatore il valore 1 corrisponde alle frasi con polarità positiva
            countPOS +=1
            POS.append(sentence) #conto le frasi con polarità positiva e poi le inserisco nella lista
    print("Ci sono", countNEG,"frasi con polarità negativa e",countPOS,"con polarità positiva,", end=" ") #formattazione per far concludere la print con uno spazio, al posto dell'accapo
    if countNEG > countPOS:
        print("quindi sono più le frasi con polarità negativa.")
    else:
        print("quindi sono più le frasi con polarità positiva.")

def main(file1, file2): #parametri del main i file che inserisco da tastiera
    file1_letto = read_file_contents(file1) #lettura primo file
    file2_letto = read_file_contents(file2) #lettura secondo file
    tokens1 = get_tokens(file1_letto) #tokenizzazione primo file
    tokens2 = get_tokens(file2_letto) #tokenizzazione secondo file
    vocabolario1 = list(set(tokens1)) #creazione vocabolario primo file
    vocabolario2 = list(set(tokens2)) #creazione vocabolario secondo file
    sentences1 = sentence_splitting(file1_letto) #sentence splitting primo file
    sentences2 = sentence_splitting(file2_letto) #sentence splitting secondo file

    print("\nProgetto LC 2023/2024 | Alessandra Bottiglieri | 648769 | Programma I")

    """ #annotazione linguistica in commento per alleggerire l'output
    print(f"ANNOTAZIONE LINGUISTICA DEL FILE {file1}:")
    annotate(sentences1) #annotazione linguistica per frase del primo file
    print("\n")
    print(f"ANNOTAZIONE LINGUISTICA DEL FILE {file2}")
    annotate(sentences2) #annotazione linguistica per frase del secondo file
    """

    print(f"\n\n1) Comparazione lunghezza tra {file1} e {file2}:")

    print(f"Il numero di frasi del file {file1} è: {len(sentences1)}")
    print(f"Il numero di frasi del file {file2} è: {len(sentences2)}")
    print(f"Il numero di token del file {file1} è: {len(tokens1)}")
    print(f"Il numero di token del file {file2} è: {len(tokens2)}")

    print(f"\n\n2) Confronto lunghezza media frasi in token dei file:")
    print(f"La lunghezza media delle frasi in token del file {file1} è: {average_sentences_length(sentences1, tokens1):.2f}")
    print(f"La lunghezza media delle frasi in token del file {file2} è: {average_sentences_length(sentences2, tokens2):.2f}")
    print(f"\nConfronto lunghezza media token in caratteri dei file:")
    print(f"La lunghezza media dei token del file {file1} è: {average_tokens_length(tokens1):.2f}")
    print(f"La lunghezza media dei token del file {file2} è: {average_tokens_length(tokens2):.2f}")

    print(f"\n\n3) Confronto numero di hapax tra i due file:")
    upper_bound500 = 500 #specifico ogni intervallo su cui analizzare la distribuzione degli hapax
    upper_bound1000 = 1000
    upper_bound3000 = 3000
    print(f"Numero di hapax del primo file {file1}:")
    print(f"Il numero di hapax per i primi {upper_bound500} token del file {file1} è: {analyze_slice(tokens1, upper_bound500):.2f}")
    print(f"Il numero di hapax per i primi {upper_bound1000} token del file {file1} è: {analyze_slice(tokens1, upper_bound1000):.2f}")
    print(f"Il numero di hapax per i primi {upper_bound3000} token del file {file1} è: {analyze_slice(tokens1, upper_bound3000):.2f}")
    print(f"Il numero di hapax per l'intero corpus di {file1} è: {count_hapax(vocabolario1)}")
    print(f"\nNumero di hapax del secondo file {file2}:")
    print(f"Il numero di hapax per i primi {upper_bound500} token del file {file2} è: {analyze_slice(tokens2, upper_bound500):.2f}")
    print(f"Il numero di hapax per i primi {upper_bound1000} token del file {file2} è: {analyze_slice(tokens2, upper_bound1000):.2f}")
    print(f"Il numero di hapax per i primi {upper_bound3000} token del file {file2} è: {analyze_slice(tokens2, upper_bound3000):.2f}")
    print(f"Il numero di hapax per l'intero corpus di {file2} è: {count_hapax(vocabolario2)}")

    print(f"\n\n4) Confronto dimensione vocabolario e TTR tra i due file:")
    print(f"La dimensione del vocabolario del file {file1} è: {len(vocabolario1)}")
    print(f"La dimensione del vocabolario del file {file2} è: {len(vocabolario2)}")
    print(f"\nType-Token Ratio del file {file1}:")
    step = 0
    while step < len(tokens1): #ciclo per porzioni incrementali di 200 token
        if step+200 > len(tokens1):
            step = len(tokens1)
        else:
            step+=200
        get_TTR_slice(tokens1, upper_bound=step) #calcolo la ricchezza lessicale per ogni porzione
    print(f"\nType-Token Ratio del file {file2}:") #stessa cosa per il secondo file
    step = 0
    while step < len(tokens2):
        if step+200 > len(tokens2):
            step = len(tokens2)
        else:
            step+=200
        get_TTR_slice(tokens2, upper_bound=step)

    print(f"\n\n5) Il numero di lemmi distinti in entrambi i file:")
    print(f"La dimensione del vocabolario dei lemmi distinti del file {file1} è: {len(set(stemming(tokens1)))}")
    print(f"La dimensione del vocabolario dei lemmi distinti del file {file2} è: {len(set(stemming(tokens2)))}")

    print(f"\n\n6) Confronto corpora dei file {file1} e {file2} per polarità positiva e negativa:")
    print(f"Distribuzione frasi polarità positiva e negativa del file {file1}")
    classifier(sentences1)
    print(f"\nDistribuzione frasi polarità positiva e negativa del file {file2}")
    classifier(sentences2)
if __name__=="__main__":
    main(sys.argv[1], sys.argv[2]) #i due argomenti che inserisco da tastiera (in questo caso i due file)