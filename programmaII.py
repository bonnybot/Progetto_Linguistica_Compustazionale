import sys
import nltk
from nltk import ngrams, bigrams, FreqDist, ConditionalFreqDist
import math
from collections import Counter

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

def pos_tagging(tokens): #PoS tagging dei token
    token_PoS = nltk.tag.pos_tag(tokens) #di default language è settato sull'inglese

    return token_PoS

def annotate(sentences): #annotazione linguistica del testo per frase (sentence splitting, tokenizzazione PoS taggata)
    for sentence in sentences:
        tokens = nltk.tokenize.word_tokenize(sentence)
        tokens_PoS = nltk.tag.pos_tag(tokens)
        print(f"La frase è: {sentence}")
        print(f"I token PoS taggati che la compongono sono: {tokens_PoS}\n")

def top50noun_adv_adj(tokens_PoS, freq): #top 50 sostantivi, avverbi e aggettivi più frequenti
    token_list = []

    for token in tokens_PoS: #itero sui token PoS taggati, considerando solo quelli con il tag che ci interessa
        if token[1]=="JJ" or token[1]=="JJR" or token[1]=="JJS" or token[1]=="NN" or token[1]=="NNS" or token[1]=="RB" or token[1]=="RBR" or token[1]=="RBS":
            freq_token = freq[token]
            tuple = (freq_token, token[0]) #creo una tupla con la relativa frequenza e il token corrispondente
            token_list.append(tuple) #inserisco in lista
    set_list = set(token_list) #escludo i duplicati
    sorted_list = sorted(set_list, reverse=True) #ordino per frequenza decrescente
    for token in sorted_list[0:50]: #stampo solo i top 50
        print(f"Frequenza: {token[0]}\tToken: {token[1]}")

def get_freq_distr(elements, n): #funzione per ottenere la frequenza di un n-gramma
    ngrammi = ngrams(elements, n)
    freq_distr = FreqDist(ngrammi)

    return freq_distr

def top20ngrams(freq_distr, top_n): #top 20 n-grammi più frequenti
    for elem, freq in freq_distr.most_common(20): #ordinati per frequenza decrescente con most_common
        print(f"Frequenza: {freq}\tToken: {elem}")
    
def top10bigramsPoS(bigramsPoS, freqBigramsPoS): #top 10 bigrammi aggettivo-sostantivo
    listbigramsPoST = []
    listFreqBigramsPoS = []
    
    for big in bigramsPoS: #itero sui bigrammi PoS taggati, considerando solo quelli col tag che ci interessa
        if (big[0][1]=="JJ" or big[0][1]=="JJR" or big[0][1]=="JJS") and (big[1][1]=="NN" or big[1][1]=="NNS"):
            freqPoS = freqBigramsPoS[big]
            tuple = (freqPoS, big) #creo una tupla con la relativa frequenza e il bigramma corrispondente
            listFreqBigramsPoS.append(tuple) #aggiungo in lista
    voc = set(listFreqBigramsPoS) #escludo i duplicati
    sortedList = sorted(voc, reverse=True) #ordino per frequenza decrescente
    for elem in sortedList[0:10]: #stampo solo i top 10
        print(f"Frequenza: {elem[0]}\tBigramma: {elem[1]}")
    
def top10bigrams_condProb(tokens, bigramsPoS): #top 10 bigrammi agg-sos per probabilità condizionata massima
    listProbBig = []
    for big in bigramsPoS: #itero sui bigrammi PoS taggati, considerando solo quelli col tag che ci interessa
        if (big[0][1]=="JJ" or big[0][1]=="JJR" or big[0][1]=="JJS") and (big[1][1]=="NN" or big[1][1]=="NNS"):
            freqBig = bigramsPoS.count(big) #calcolo la frequenza del bigramma
            elementfreq1 = tokens.count(big[0][0]) #calcolo la frequenza del primo elemento del bigramma
            conditionalProb = freqBig/elementfreq1 #frequenza del bigramma fratto la frequenza del primo elemento
            tuple = (conditionalProb, big) #creo una tupla con la relativa probabilità condizionata e il bigramma corrispondente
            listProbBig.append(tuple) #aggiungo in lista
    voc = set(listProbBig) #escludo i duplicati
    sortedList = sorted(voc, reverse=True) #ordino per probabilità condizionata massima
    for elem in sortedList[0:10]: #stampo solo i primi 10
        print(f"Probabilità condizionata: {elem[0]}\tBigramma: {elem[1]}")

def top10bigrams_jointProb(tokens, bigramsPoS): #top 10 bigrammi agg-sos per probabilità congiunta massima
    listProbBig = []
    for big in bigramsPoS: #stessa cosa della precedente funzione
        if (big[0][1]=="JJ" or big[0][1]=="JJR" or big[0][1]=="JJS") and (big[1][1]=="NN" or big[1][1]=="NNS"):
            freqBig = bigramsPoS.count(big)
            elementfreq1 = tokens.count(big[0][0])
            conditionalProb = freqBig/elementfreq1
            probElement1 = elementfreq1/len(tokens) #calcolo la probabilità del 1° elemento
            jointProb = conditionalProb*probElement1 #calcolo la prob congiunta, moltiplicando la prob condizionata per la prob del 1° elem
            tuple = (jointProb, big) #creo una tupla con relativa prob congiunta e il bigramma corrispondente
            listProbBig.append(tuple) #aggiungo in lista
    voc = set(listProbBig) #escludo i duplicati
    sortedList = sorted(voc, reverse=True) #ordino per prob congiunta massima
    for elem in sortedList[0:10]: #stampo solo i primi 10
        print(f"Probabilità congiunta: {elem[0]:.4f}\tBigramma: {elem[1]}") #approssimazione del numero decimale

def top10bigrams_MI_LMI_common(tokens, bigramsPoS): #top 10 bigrammi per MI e LMI massima
    listMIBig = []
    listLMIBig = []
    listCommon = []
    onlyBigMI = []
    onlyBigLMI = []
    for big in bigramsPoS: #anche qui considero solo quelli con il giusto tag
        if (big[0][1]=="JJ" or big[0][1]=="JJR" or big[0][1]=="JJS") and (big[1][1]=="NN" or big[1][1]=="NNS"):
            freqBig = bigramsPoS.count(big) #frequenza del bigramma
            freqt1 = tokens.count(big[0][0]) #frequenza del 1° token
            freqt2 = tokens.count(big[1][0]) #frequenza del 2° token
            MI = math.log((freqBig*len(tokens)/(freqt1*freqt2)), 2) #MI = log_2 del prodotto della freq del big per il corpus, fratto il prodotto tra le freq dei token del big
            tupleMI = (MI, big) #creo una tupla con la relativa MI e il bigramma corrispondente
            listMIBig.append(tupleMI) #aggiungo in lista
            LMI = freqBig*MI #Local Mutual Information = freq del bigramma per la MI
            tupleLMI = (LMI, big) #creo tupla con relativa LMI e bigramma corrispondente
            listLMIBig.append(tupleLMI) #aggiungo in lista
    vocMI = set(listMIBig) #escludo i duplicati dalla lista dei bigrammi con MI
    sortedListMI = sorted(vocMI, reverse=True) #ordino per MI massima
    for elem in sortedListMI[0:10]:
        tOnlyBigMI = (elem[1][0], elem[1][1]) #creo una tupla solo col bigramma PoS taggato per il confronto finale
        onlyBigMI.append(tOnlyBigMI) #inserisco la tupla in lista
    print("\nd. Ordinati per Mutual Information massima:")
    for elemMI in sortedListMI[0:10]: #stampo solo i primi 10
        print(f"Mutual Information: {elemMI[0]:.2f}\tBigramma: {elemMI[1]}")
    vocLMI = set(listLMIBig) #escludo i duplicati dalla lista dei bigrammi con LMI
    sortedListLMI = sorted(vocLMI, reverse=True) #ordino per LMI massima
    for elem in sortedListLMI[0:10]: 
        tOnlyBigLMI = (elem[1][0], elem[1][1]) #anche qui tupla solo col bigramma PoS taggato
        onlyBigLMI.append(tOnlyBigLMI)
    print("\ne. Ordinati per Local Mutual Information massima:")
    for elemLMI in sortedListLMI[0:10]: #stampo solo i primi 10
        print(f"Local Mutual Information: {elemLMI[0]:.2f}\tBigramma: {elemLMI[1]}")
    print("\nf. Elementi comuni tra top 10 MI e LMI:")
    vocBigMI = set(onlyBigMI) #escludo i duplicati 
    vocBigLMI = set(onlyBigLMI)
    sortedBigMI = sorted(vocBigMI, reverse = True) #riordino le liste
    sortedBigLMI = sorted(vocBigLMI, reverse=True)
    for elemMI in sortedBigMI[0:10]: #confronto finale per eventuali bigrammi comuni 
        for elemLMI in sortedBigLMI[0:10]: #cicli for annidati per iterare su entrambe le liste
            if elemMI == elemLMI: #confronto liste un elemento per volta
                listCommon.append(elemMI) #aggiungo in lista se c'è un elemento in comune
    vocCommon = set(listCommon)
    sortedListCommon = sorted(vocCommon, reverse=True) #ordinamento decrescente
    count = 1
    if not sortedListCommon: #controllo se la lista è vuota
        print("Non c'è nessun elemento in comune.")
    else:
        for common in sortedListCommon: #se ci sono elementi comuni, ne stampo il numero con relativo elemento
            print(f"{count}° elemento comune: {common}")
            count+=1

def markov2(sentences, tokens): #calcolo probabilità Markov 2° ordine
    probabilities = []
    bigrams = list(nltk.bigrams(tokens)) #creo lista bigrammi
    trigrams = list(nltk.trigrams(tokens)) #creo lista trigrammi
    tokensFreq = FreqDist(tokens) #frequenza dei singoli token
    bigramsFreq = ConditionalFreqDist(bigrams) #frequenza dei bigrammi
    trigramsFreq = ConditionalFreqDist([((t1, t2), t3) for t1, t2, t3 in trigrams]) #frequenza dei trigrammi

    for sentence in sentences: #iterazione su tutte le frasi del testo
        tokenSentence = nltk.word_tokenize(sentence) #sentence splitting 
        for i in range(len(tokenSentence)): #itero su tutta la frase corrispondente
            if i == 0: #primo token
                sentenceProb = (tokensFreq[tokenSentence[0]])/(len(tokens)) #frequenza del 1° token fratto dimensione corpus
            if i == 1: #secondo token
                prob = (bigramsFreq[tokenSentence[i-1]][tokenSentence[i]])/(tokensFreq[tokenSentence[0]]) #freq del bigramma fratto token precedente
                sentenceProb = sentenceProb * prob
            if i>1: #dal terzo token
                prob = (trigramsFreq[(tokenSentence[i-2], tokenSentence[i-1])][tokenSentence[i]])/(bigramsFreq[tokenSentence[i-2]][tokenSentence[i-1]]) #freq trigramma fratto freq bigramma
                sentenceProb = sentenceProb * prob
    probabilities.append((sentence, sentenceProb)) #aggiungo in lista la relativa frase con probabilità corrispondente
    maxProb = sorted(probabilities, reverse=True) #ordinamento decrescente per massima probabilità
    print(f"{maxProb[0][0]}")  #stampo solo la frase, se volessi stampare anche la probabilità {maxProb[0][1]}

def sentences10or20token(sentences, tokens):
    sentences10or20t = []
    tokenFreq = FreqDist(tokens)
    
    for sentence in sentences:
        sentence_tokens = nltk.tokenize.word_tokenize(sentence)
        if len(sentence_tokens)>=10 & len(sentence_tokens)<=20: #frase compresa tra 10 e 20 token
            count = 0
            for token in sentence_tokens:
                if tokenFreq[token]>=2: #controllo che il token non sia un hapax
                    count+=1
            if count >= len(sentence_tokens) // 2: #controllo che almeno la metà dei token della frase non sia un hapax
                sentences10or20t.append(sentence) #aggiungo la frase in lista se rispetta tutti i requisiti
    freqTokens = [FreqDist(nltk.word_tokenize(sentence)) for sentence in sentences10or20t] #calcolo distribuzione di frequenza per ogni token nelle frasi selezionate
    avg = [sum(freq.values())/len(freq) for freq in freqTokens] #calcolo della frequenza media per ogni token
    maxAvg = sentences10or20t[avg.index(max(avg))] #essendo una lista, cerco l'indice dell'elemento di cui prendo il massimo valore
    print("\na. Frase con la media di distribuzione frequenza dei token più alta:")
    print(f"{maxAvg}") 
    minAvg = sentences10or20t[avg.index(min(avg))] #stessa cosa per il valore minimo
    print("\nb. Frase con la media di distribuzione frequenza dei token più bassa:")
    print(f"{minAvg}")
    print("\nc. Frase con probabilità più alta seconda modello di Markov 2° ordine:")
    markov2(sentences, tokens)
        
def get_NE(tokensPos):
    NE_tree = nltk.ne_chunk(tokensPos) #creo l'albero di entità nominate con i tokens PoS taggati
    NE = {} #creo il dizionario per le entità nominate
    for nodo in NE_tree: #itero su ogni nodo dell'albero
        if hasattr(nodo, 'label'): #se il nodo ha un attributo label, quindi il nome dell'entità
            if nodo.label() not in NE:
                NE[nodo.label()] = Counter() #per contare il numero di occorrenze
            NE[nodo.label()][nodo[0][0]]+=1 #incremento il conteggio per l'entità nominata corrente
    for label, counts in NE.items(): #itero sia sulle chiavi che sui valori del dizionario per stampare entità e valori
        print(f"\nPer la classe {label} i primi 15 (ove presenti) per frequenza decrescente:")
        for entity, count in counts.most_common(15): #stampo i primi 15 in ordine decrescente
            print(f"Token: {entity} | Frequenza: {count}")


def main (file):
    print("\nProgetto LC 2023/2024 | Alessandra Bottiglieri | 648769 | Programma II")

    file_letto = read_file_contents(file) #leggo il file
    sentences = sentence_splitting(file_letto) #sentence splitting del testo
    tokens = get_tokens(file_letto) #tokenizzo
    tokens_PoS = nltk.tag.pos_tag(tokens) #PoS tagging dei token
    PoS_list = [PoS for token, PoS in tokens_PoS] #list comprehension per avere solo i tag

    """ #annotazione linguistica in commento per alleggerire l'output
    print(f"ANNOTAZIONE LINGUISTICA FILE {file}")
    annotate(sentences) #sentence splitting e tokenizzazione PoS taggata
    """

    freq_PoS = nltk.FreqDist(tokens_PoS) #frequenza dei token PoS taggati
    print(f"\n\n1) Top 50 sostantivi, avverbi e aggettivi più frequenti del file {file}:\n")
    top50noun_adv_adj(tokens_PoS, freq_PoS)

    print(f"\n\n2) Top 20 n-grammi più frequenti del file {file}:")
    for n in range(1, 6): #n-grammi da 1 a 5
        freq_distr = get_freq_distr(tokens, n) #calcolo frequenza n-gramma
        print(f"\nTop 20 {n}-grammi più frequenti")
        top20ngrams(freq_distr, top_n=5)

    print(f"\n\n3) Top 20 n-grammi PoS taggati più frequenti del file {file}:")
    for n in range(1, 4): #n-grammi da 1 a 3
        freq_distr = get_freq_distr(tokens_PoS, n) #calcolo frequenza n-gramma
        print(f"\nTop 20 {n}-grammi più frequenti")
        top20ngrams(freq_distr, top_n=5)

    print(f"\n\n4) Top 10 bigrammi composti da Aggettivo e Sostantivo del file {file}:")
    bigramsPoS = list(bigrams(tokens_PoS)) #creo lista bigrammi
    freqBigramsPoS = FreqDist(bigramsPoS) #calcolo frequenza bigrammi
    print(f"\na. Ordinati per frequenza decrescente:")
    top10bigramsPoS(bigramsPoS, freqBigramsPoS)

    print("\nb. Ordinati per probabilità condizionata massima:")
    top10bigrams_condProb(tokens, bigramsPoS)

    print("\nc. Ordinati per probabilità congiunta massima:")
    top10bigrams_jointProb(tokens, bigramsPoS)

    top10bigrams_MI_LMI_common(tokens, bigramsPoS)

    print(f"\n\n5) Frasi del file {file} con lunghezza tra 10 e 20 token, di cui almeno la metà non è un hapax:")
    sentences10or20token(sentences, tokens)

    print(f"\n\n6) Entità nominate del file {file}:")
    get_NE(tokens_PoS)
    

if __name__=="__main__":
    main(sys.argv[1]) #argomento che inserisco da tastiera, in questo caso il nome del file
    