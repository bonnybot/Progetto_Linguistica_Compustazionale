# 📌 Linguistica Computazionale - Progetto 2023/2024

## 📖 Descrizione
Questo progetto di **Linguistica Computazionale** implementa due programmi in Python per l'analisi linguistica e statistica di corpora testuali in inglese. Utilizzando la libreria NLTK, il progetto analizza la struttura linguistica, la ricchezza lessicale e la polarità testuale. Tecnologie: Python, NLTK.

Il progetto si compone di due programmi principali:
1. **Programma I:** Confronto tra due corpora sulla base di metriche linguistiche e statistiche.
2. **Programma II:** Estrazione di informazioni linguistiche avanzate da un corpus.

## 🚀 Tecnologie Utilizzate
- **Python**
- **NLTK** (Natural Language Toolkit)
- **Modello di classificazione del sentiment** (`sentiment-classifier.pkl`)

---

## 🔧 Installazione
1. **Clonare il repository**
   ```bash
   git clone https://github.com/tuo-username/Progetto_Linguistica_Computazionale.git
   cd linguistica-computazionale
   ```

2. **Creare un ambiente virtuale (opzionale ma consigliato)**
   ```bash
   python -m venv env
   source env/bin/activate  # Mac/Linux
   env\Scripts\activate  # Windows
   ```

3. **Installare le dipendenze**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Struttura del Progetto
```
📁 Progetto_Linguistica_Computazionale
 ├── 📄 programmaI.py       # Analisi comparativa dei corpora
 ├── 📄 programmaII.py      # Estrazione di informazioni linguistiche
 ├── 📄 jobs.txt            # Corpora di testo (UTF-8)
 ├── 📄 metamorphosis.txt  # Corpora di testo (UTF-8)
 ├── 📄 output1.txt        # Output del Programma I
 ├── 📄 output2jobs.txt    # Output del Programma II per il corpus "jobs.txt"
 ├── 📄 output2metamorphosis.txt  # Output del Programma II per il corpus "metamorphosis.txt"
 ├── 📄 sentiment-classifier.pkl  # Modello di classificazione del sentiment
 ├── 📄 README.md          # Questo file
```

---


## 🖥️ Utilizzo
### **Programma I - Analisi Comparativa**
Eseguire il comando:
```bash
python programmaI.py jobs.txt metamorphosis.txt
```
**Output:** File `output1.txt` con le seguenti informazioni:
- Numero di frasi e token
- Lunghezza media delle frasi e dei token
- Hapax Legomena
- Type-Token Ratio (TTR)
- Vocabolario dei lemmi
- Distribuzione di frasi con polarità positiva e negativa
- **Classificazione della polarità delle frasi utilizzando il modello `sentiment-classifier.pkl`**

### **Programma II - Estrazione Linguistica**
Eseguire il comando per il corpus "jobs.txt":
```bash
python programmaII.py jobs.txt
```
Eseguire il comando per il corpus "metamorphosis.txt":
```bash
python programmaII.py metamorphosis.txt
```
**Output:**
- `output2jobs.txt` per il corpus "jobs.txt"
- `output2metamorphosis.txt` per il corpus "metamorphosis.txt"

Contiene:
- I 50 sostantivi, aggettivi e avverbi più frequenti
- I 20 n-grammi più comuni
- I 20 n-grammi di Part-of-Speech più comuni
- I 10 bigrammi aggettivo-sostantivo più rilevanti secondo diverse metriche
- Analisi di frasi sulla base di distribuzione di frequenze e modello di Markov
- Estrazione delle Named Entities più frequenti

---

## 📝 Autore
👤 **[Alessandra Bottiglieri](www.linkedin.com/in/alessandra-bottiglieri-2a6916177)**
