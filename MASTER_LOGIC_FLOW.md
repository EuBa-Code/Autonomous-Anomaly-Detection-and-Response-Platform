# 🧠 MASTER LOGIC FLOW: Cronaca di un Sistema MLOps

> **Introduzione al Documento**
>
> Questo non è un manuale utente. Questo è il racconto dettagliato di **come vive un dato** all'interno del nostro sistema di Anomaly Detection per lavatrici industriali.
>
> Abbiamo diviso questa storia in "Atti", seguendo il ciclo di vita naturale dell'informazione: dalla sua nascita (simulata), alla sua maturazione (features), al suo utilizzo per l'addestramento (training), fino al suo destino finale nel mondo reale (streaming & inferenza).
>
> Ogni sezione risponde a tre domande fondamentali:
> 1.  **Cosa succede?** (L'evento tecnico)
> 2.  **Come succede?** (La meccanica interna)
> 3.  **Perché succede così?** (La motivazione architetturale)

---

### Mappa dei Componenti

| Componente | Ruolo | Alternativa Scartata (e perché) |
| :--- | :--- | :--- |
| **PySpark** | Generazione e processamento massivo di dati | **Pandas** (Non scala su milioni di righe, troppa RAM) |
| **Parquet** | Storage efficiente e tipizzato su disco | **CSV** (Lento, perde i tipi, occupa troppo spazio) |
| **Isolation Forest** | Modello Anomaly Detection Unsupervised | **XGBoost/RandomForest** (Richiedono etichette "Guasto" che non abbiamo) |
| **MLflow** | Tracciamento esperimenti e Model Registry | **WandB / Neptune** (MLflow è open source e self-hosted facilmente) |
| **Feast** | Feature Store (Ponte Batch-Streaming) | **DB Custom** (Feast gestisce nativamente la Point-in-Time correctness) |
| **Redis** | Online Store per Feature a bassa latenza | **PostgreSQL** (Troppo lento per query < 10ms ad alta concorrenza) |
| **Redpanda** | Message Broker per Streaming (Kafka-API) | **Apache Kafka** (Redpanda è in C++, senza JVM, molto più leggero) |
| **Quix Streams** | Elaborazione streaming stateful (Rolling Windows) | **Spark Streaming** (Troppo pesante e lento per latenze sub-secondo) |
| **FastAPI** | API Gateway per l'Inferenza | **Flask/Django** (Meno performanti e senza validazione Pydantic nativa) |

### Il Flusso Dati End-to-End

```ascii
```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                        ARCHITETTURA MLOps - ANOMALY DETECTION                            ║
║                                                                                          ║
║  FASI TEMPORALI:                                                                         ║
║  [FASE 1] Setup iniziale  →  [FASE 2] Training  →  [FASE 3] Runtime (parallelo)          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [FASE 1]  SETUP INIZIALE  —  Eseguita UNA VOLTA prima di tutto il resto  (make setup)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────────┐       ┌──────────────────────────────────────────┐
  │      CREATE DATASETS         │       │           HIST. INGESTION                │
  │        (PySpark)             │──────▶│           (Spark Windows)                │
  │                              │       │                                          │
  │ • Simula 10 macchine         │       │ • Calcola Rolling Mean (10min)           │
  │ • Mesi di storico sintetico  │       │ • Calcola Rolling Std  (10min)           │
  │ • Inietta anomalie           │       │ • NON scala (mantiene mm/s, A, °C)       │
  │   etichettate (ground truth) │       │                                          │
  └──────────────┬───────────────┘       └──────────────────┬───────────────────────┘
                 │                                          │
                 ▼                                          ▼
  ┌──────────────────────────┐       ┌──────────────────────────────────────────────┐
  │      [Parquet Raw]       │       │               [Parquet Feat]                 │
  │       Dati Grezzi        │       │           ◀══ OFFLINE STORE ══▶              │
  │                          │       │                                              │
  │ • Valori crudi           │       │ • Stesse righe + statistiche temporali       │
  │ • Scala originale        │       │ • Es: Vibrazione: 50 mm/s                    │
  │   (mm/s, A, °C)          │       │       Vibration_mean10m: 48 mm/s             │
  │ • Nessuna feature        │       │       Vibration_std10m:   2 mm/s             │
  │   calcolata              │       │ • NON normalizzato                           │
  └──────────────────────────┘       └──────────────────┬───────────────────────────┘
                                                        │
                                                        ▼
                                     ┌──────────────────────────────────────────────┐
                                     │            [Feast Materialize]               │
                                     │                                              │
                                     │ • Legge Offline Store (Parquet Feat)         │
                                     │ • Copia SOLO l'ultima riga per ogni          │
                                     │   macchina dentro Redis                      │
                                     │ • Scopo: risolvere Cold Start                │
                                     │   (Redis non parte vuoto)                    │
                                     └──────────────────┬───────────────────────────┘
                                                        │
                                                        ▼ (popola Redis per la prima volta)
                                                    [ Redis ]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [FASE 2]  TRAINING  —  Batch job, gira dopo il setup (o periodicamente)  (make train)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Legge da [Parquet Feat]
        │
        ▼
  ┌─────────────────────────────────────┐       ┌───────────────────────────────────┐
  │         TRAINING SERVICE            │       │             MLflow                │
  │    (IsoForest + StandardScaler)     │──────▶│           (Registry)              │
  │                                     │       │                                   │
  │ 1. Fit StandardScaler               │ logga │ • Versioning (v1, v2...)          │
  │    (calcola μ e σ per colonna)      │       │ • Staging → Production            │
  │ 2. Salva preprocessor.joblib        │ model │   (promozione manuale o script)   │
  │ 3. Trasforma: z = (x - μ) / σ       │       │ • Confronto esperimenti           │
  │ 4. Fit IsoForest su dati scalati    │ metr. │   (accuracy, precision, recall)   │
  │ 5. Valuta su test set               │       │ • Salva artifacts:                │
  │ 6. Registra modello su MLflow       │ param │   model.pkl + preprocessor.joblib │
  └─────────────────────────────────────┘       └───────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [FASE 3]  RUNTIME  —  Questi 3 blocchi girano IN PARALLELO  (make streaming)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                                                    ┌───────────────────────┐
                                                    │    MLflow Registry    │
                                                    │                       │
                                                    │  • model.pkl          │
                                                    │  • preprocessor.joblib│
                                                    └───────────┬───────────┘
                                                                │
                                                                │ (1) all'avvio dell'API,
                                                                │     scarica modello
                                                                │     e preprocessor
                                                                │     in RAM (una volta sola)
                                                                │
                                                                ▼
                              ┌──────────────────────────────────────────────────────┐
                              │                    Redis                             │
                              │                 (Online Store)                       │
                              │                                                      │
                              │  • Aggiornato ogni < 1 secondo da Quix (streaming)  │
                              │  • Letto in < 1ms dall'Inference Service            │
                              │  • Valori NON scalati (scala originale mm/s, A, °C) │
                              └──────────────────┬───────────────────────────────────┘
                                       ▲         │
                                       │         │
               (2) Quix scrive         │         │  (3) Inference legge
               features aggiornate     │         │      features per predizione
               ogni < 1 secondo        │         │
                                       │         ▼
  ┌────────────────────────────┐       │       ┌──────────────────────────────────────┐
  │      STREAMING PIPELINE    │       │       │         INFERENCE SERVICE            │
  │                            │       │       │            (FastAPI)                 │
  │  ┌──────────────────────┐  │       │       │                                      │
  │  │   PRODUCER SERVICE   │  │       │       │  Per ogni richiesta HTTP:            │
  │  │    (Simulazione)     │  │       │       │                                      │
  │  │                      │  │       │       │  1. Riceve machine_id                │
  │  │ • Genera letture     │  │       │       │  2. Legge features da Redis          │
  │  │   live ogni secondo  │  │       │       │  3. Scala: z = (x - μ) / σ          │
  │  │ • Simula normale     │  │       │       │     (usa preprocessor in RAM)        │
  │  │   e anomalie         │  │       │       │  4. Passa vettore a IsoForest        │
  │  └──────────┬───────────┘  │       │       │     (usa model in RAM)               │
  │             │              │       │       │  5. Restituisce JSON                 │
  │             ▼              │       │       │                                      │
  │       [Redpanda]           │       │       │  {"is_anomaly": true,                │
  │       Message Broker       │       │       │   "machine_id": 1,                   │
  │                            │       │       │   "anomaly_score": -0.42,            │
  │  • Ordine FIFO garantito   │       │       │   "confidence": 0.95,                │
  │  • Bufferizza se Quix      │       │       │   "timestamp": "2026-02-18..."}      │
  │    è lento o offline       │       │       │                                      │
  │  • Persiste su disco       │       │       └──────────────────────────────────────┘
  │  • Compatibile Kafka API   │       │
  │             │              │       │
  │             ▼              │       │
  │     [Quix Streams]         │───────┘
  │     + RocksDB (stato)      │
  │                            │
  │  • Ascolta Redpanda        │
  │  • RocksDB mantiene stato  │
  │    per finestra mobile     │
  │  • Calcola Rolling Mean    │
  │    e Std su dati in arrivo │
  │  • Scrive su Redis         │
  │    ogni < 1 secondo        │
  └────────────────────────────┘


LEGENDA:
──▶   Flusso dati
[ ]   Storage / Infrastruttura
┌┐    Servizio / Container Docker
▲▼    Direzione del flusso
(1)(2)(3)  Ordine logico delle operazioni in Fase 3
μ     Media calcolata da StandardScaler
σ     Deviazione standard calcolata da StandardScaler
```
```

---

# ATTO PRIMO: LA GENESI (Generazione Dati Sintetici)

Tutto inizia nel container `create_datasets_service`.
Siamo in uno scenario industriale simulato. Non abbiamo sensori fisici collegati a motori reali, quindi dobbiamo crearne una copia digitale credibile.

### 1.1 Il Motore Fisico Simulato (PySpark Rule-Based)
Al contrario di quanto farebbe un principiante usando numeri casuali puri (`random.uniform`), il nostro generatore segue un set di regole fisiche rigorose implementate in **PySpark**.
Abbiamo scelto PySpark non a caso: dovevamo generare milioni di righe (mesi di storico) per 10 macchine diverse. Python puro (Pandas) sarebbe stato lento e avrebbe consumato tutta la RAM. PySpark distribuisce il calcolo, permettendoci di scalare a miliardi di righe se necessario.

**La Logica "Rule-Based" al posto di SDV**:
Inizialmente avevamo considerato l'uso di Generative AI (SDV/Gaussian Copula), ma abbiamo optato per un approccio **Deterministico con Rumore**. Perché? Perché conosciamo le leggi della fisica di una lavatrice meglio di quanto un'AI possa impararle da pochi dati.
*   **Fasatura del Ciclo**: Una lavatrice ha stati precisi (Idle, Fill, Heat, Wash, Spin, Drain).
*   **Correlazione Causale Imposta**:
    *   Se lo stato è `Spin` (Centrifuga), il motore *deve* girare a 1400 RPM e la vibrazione *deve* essere alta.
    *   Se lo stato è `Heat` (Riscaldamento), la corrente *deve* essere alta (35 Ampere per le resistenze) e la temperatura dell'acqua *deve* salire.
*   **Injection di Rumore (`randn`)**: Per non rendere i dati "finti" e troppo perfetti, aggiungiamo rumore gaussiano a ogni valore. Una corrente di 35A diventa 34.8A, 35.2A, ecc. Questo simula le fluttuazioni reali dei sensori.
*   **Anomalie Sintetiche**: Iniettiamo guasti con logica precisa (es. "Se è un'anomalia di tipo Overheating, porta la temperatura a 90°C durante il lavaggio").
Questa scelta ci dà il **controllo totale** sulla "Ground Truth": sappiamo esattamente quando e perché una macchina è rotta, cosa impossibile con dati reali non etichettati o generati da "black box" AI.

### 1.2 La Cristallizzazione (Parquet Storage)
Una volta generato questo enorme DataFrame in memoria, dobbiamo salvarlo.
Non usiamo CSV. Usiamo **Parquet**.
Immagina il CSV come un libro scritto riga per riga. Se vuoi cercare solo una parola specifica (es. "Temperatura"), devi leggere tutto il libro.
Parquet è come un'enciclopedia indicizzata per argomenti (Colonne). Se l'ingestione successiva ha bisogno solo della colonna `Timestamp` e `Vibration`, Parquet permette di leggere *solo* quei bit dal disco, ignorando terabyte di dati inutili (es. `Current_L1`, `Voltage`).
Inoltre, Parquet mantiene i tipi di dato nativi. Un `float32` resta un `float32`. Non c'è il rischio che "10.00" venga letto come la stringa "10.00" e causi errori di calcolo.

---

# ATTO SECONDO: LA RAFFINERIA (Feature Engineering)

Il dato grezzo ("Vibrazione: 50") è solo un numero. Non porta informazione sufficiente. Il servizio `hist_ingestion_service` si occupa di trasformare questi numeri in **Conoscenza** (Features).

### 2.1 La Memoria Temporale (Spark Window Functions)
Un valore di vibrazione alto è grave? Dipende. È un picco di un secondo o dura da un'ora?
Per capirlo, usiamo le **Window Functions** di Spark.
Creiamo una "finestra mobile" che guarda indietro nel tempo (agli ultimi 10 minuti di dati per quella specifica macchina).
Calcoliamo statistiche aggregate:
*   `rolling_mean`: La media mobile. Se la media è bassa e il valore attuale è alto, è un picco.
*   `rolling_std`: La deviazione standard. Ci dice quanto il segnale è "nervoso".
Questo passaggio trasforma ogni riga da un evento isolato a un **riassunto storico**. Il modello non vedrà solo "ora", ma "ora rispetto a prima".

### 2.2 Il Risultato Intermedio (Parquet)
Una volta calcolate le medie mobili, salviamo tutto in **Parquet**.
**Nota Bene**: A questo stadio i dati sono **ancora nelle loro unità di misura originali** (mm/s, Ampere, °C).
Non abbiamo ancora normalizzato (niente StandardScaler).
Perché? Perché il Feature Store deve contenere i dati "fisici" reali. La decisione di come scalarli (es. MinMax vs Standard) spetta al Data Scientist nel momento del training, non all'ingegnere dei dati.

---

# ATTO TERZO: L'APPRENDIMENTO (Training & MLflow)

Il container `training_service` si sveglia. Trova i dati puliti e il preprocessor. È ora di imparare.

### 3.1 La Normalizzazione (StandardScaler) & Training
Il training service ha due compiti prima di addestrare:
1.  **Fit Scaler**: Legge i dati grezzi dal Parquet e calcola Media ($\mu$) e Deviazione Standard ($\sigma$) per ogni colonna.
2.  **Salvataggio**: Salva queste statistiche in `preprocessor.joblib`. Questo file è cruciale: sarà l'unico "arbitro" per decidere cosa è normale in futuro.
3.  **Transform**: Applica la formula $z = (x - \mu) / \sigma$ ai dati di training. Ora Vibrazione e Corrente sono confrontabili (entrambi oscillano intorno a 0).

### 3.2 Unsupervised Learning (Isolation Forest)
Perché usiamo **Isolation Forest**?
Nelle fabbriche reali, i guasti sono rari ("anomalie"). Abbiamo milioni di righe "Normali" e pochissime "Guaste". Inoltre, spesso non sappiamo *quali* siano quelle guaste (nessuno le ha etichettate).
Non possiamo usare un classificatore classico (Random Forest, XGBoost) perché:
1.  Il dataset è sbilanciato (Imbalanced).
2.  Non abbiamo etichette affidabili (Unlabeled).
L'Isolation Forest (IsoForest) è geniale: non cerca di capire cosa è "guasto". Cerca di capire cosa è "diverso".
Immagina di tagliare a caso lo spazio dei dati. I punti "normali" sono tutti vicini, ammassati in un cluster denso. Per isolare un punto normale, devi fare tanti tagli.
I punti "anomali" sono rari e lontani dal gruppo. Basta un taglio o due per isolarli.
L'algoritmo conta i tagli. Pochi tagli = Anomalia. Tanti tagli = Normalità. Semplice, veloce, efficace.

### 3.3 Il Diario di Bordo (MLflow) & Model Promotion
Mentre il modello impara, parla con **MLflow**.
Non ci limitiamo a stampare metriche a video. Le inviamo a un server centrale.
*   **Tracking**: Salviamo iperparametri (`n_estimators`, `contamination`) e metriche (`accuracy`, `precision`). Questo ci permette di confrontare esperimenti diversi nel tempo ("Il modello di oggi è migliore di quello di ieri?").
*   **Artifacts**: Salviamo il modello fisico (`model.pkl`) e il preprocessor (`preprocessor.joblib`) dentro MLflow.
*   **Model Registry & Promotion**:
    *   Ogni nuovo training registra una nuova versione (`v1`, `v2`, `v3`) nel Model Registry.
    *   Di default, il nuovo modello entra in stadio **None** o **Staging**.
    *   **La Promozione a Production**: Attualmente è un passaggio **manuale** (o via script di CD). Un umano (o un test automatico rigoroso) guarda le metriche su MLflow UI. Se l'accuratezza è superiore al modello precedente, clicca su "Transition to Production". L'Inference Service scaricherà sempre e solo l'ultima versione marcata come "Production".

---

# ATTO QUARTO: IL PONTE (Feast Feature Store)

Abbiamo un modello addestrato (Offline). Ora dobbiamo preparare il sistema per gestire i dati in tempo reale (Online).
Qui entra in gioco **Feast**.
Feast risolve il dualismo tra storage lento (Parquet) e veloce (Redis).

### 4.1 La Definizione Unica
Definiamo le features in codice Python (`feature_store/feature_definitions.py`). Questa definizione è l'unica fonte di verità. Dice: "La feature `Vibration_rollingMean` è un Float32 e si trova in questa colonna del Parquet".

### 4.2 La Materializzazione (Data Sync)
Il problema classico è che i dati offline (File Parquet) sono lenti da leggere. Ma come si "parlano" con lo streaming?
1.  **I Dati Storici (Offline)**: Vengono processati da Spark (`hist_ingestion_service`).
    *   **Preprocessing**: Sì, anche a loro vengono applicate le Rolling Windows (es. media ultimi 10 min), MA **non** lo StandardScaler.
    *   **Perché non scalati?**: Nel Feature Store salviamo i valori "fisici" (es. 50 mm/s), non quelli astratti (es. 1.2 sigma). La standardizzazione è parte del modello, non del dato.
    *   Vengono salvati in Parquet.
    *   Servono per **Addestrare il Modello**.
2.  **Il Trasloco (Materialize)**: Eseguiamo il comando `feast materialize`.
    *   Feast legge questi dati storici puliti dal Parquet.
    *   Copia i valori più recenti dentro **Redis** (Online Store).
    *   **A che serve?**: Serve per il "Cold Start". Se accendiamo il sistema ora, Redis non è vuoto, ma ha l'ultimo stato noto dal passato.
3.  **I Dati Streaming (Online)**: Arrivano da Kafka/Redpanda.
    *   Vengono processati da Quix (che calcola le Rolling Windows).
    *   Vengono scritti su Redis **ancora in scala originale** (es. mm/s).

Quindi Redis è il punto di incontro: contiene dati che arrivano sia dal passato (Batch/Materialize) sia dal presente (Streaming).
Ma attenzione: sono dati **Feature Engineered** (hanno la media mobile) ma **NON ancora Scalati**.
Chi li scala? L'Inference Service, un attimo prima di passarli al modello, usando il `preprocessor.joblib` caricato all'avvio.
    *   *Attenzione*: In Redis non copia tutto lo storico. Copia solo **l'ultima fotografia valida** per ogni macchina.
    *   *Perché?* Perché quando l'API chiederà "come sta la macchina 1?", Redis dovrà rispondere in 1 millisecondo. Non può mettersi a cercare in file giganti.

### 4.3 Deep Dive: Ma se c'è lo Streaming, a che serve la Materializzazione?
Hai ragione a chiedertelo. Se lo streaming (`Quix`) aggiorna Redis ogni secondo, perché dobbiamo copiare i dati vecchi dal Parquet?
Ci sono due motivi fondamentali:

1.  **Cold Start (Avvio a Freddo)**:
    Immagina di accendere il sistema di streaming *adesso*.
    Per i primi 10 minuti, lo streaming non ha abbastanza dati in memoria per calcolare una "media mobile a 10 minuti". Partirebbe da zero o darebbe `NaN`.
    La Materializzazione serve a "pre-caricare" Redis con l'ultimo stato valido calcolato ieri notte (Batch). Quando lo streaming parte, trova già un valore sensato in Redis.

2.  **Backfill (Correzione del Passato)**:
    Il **Backfill** è l'atto di "riempire all'indietro" i buchi o correggere gli errori.
    Mettiamo che scopriamo un bug nel codice di streaming (es. la formula della media era sbagliata per 3 giorni).
    Non possiamo tornare indietro nel tempo dello streaming.
    Ma possiamo:
    *   Correggere il codice e rilanciare il calcolo sui dati storici (Batch/Spark).
    *   Sovrascrivere Redis con i dati corretti (`Materialize`).
    *   In questo modo, abbiamo "guarito" il sistema retroattivamente.

---

# ATTO QUINTO: IL TEMPO REALE (Streaming con Redpanda & Quix)

Il training è finito. Il sistema è pronto. Accendiamo l'interruttore della simulazione (`make streaming`).
Il tempo inizia a scorrere.

### 5.1 Il Produttore (Producer Service)
Un piccolo script Python inizia a generare dati sintetici "live".
Non li scrive su disco. Li invia via rete a **Redpanda**.

### 5.2 Il Tubo (Redpanda)
**Redpanda** è la nostra spina dorsale. È un message broker compatibile con Kafka ma infinitamente più leggero (niente JVM).
Il suo ruolo è garantire che nessun dato vada perso.
Se il servizio che deve elaborare i dati (Consumer) si rompe o è lento, Redpanda accumula i messaggi (Buffering) e li tiene al sicuro su disco finché il consumatore non torna online. Garantisce l'ordine cronologico (First-In, First-Out) e la persistenza.

### 5.3 Il Cervello in Movimento (Quix Streams)
Qui avviene la magia dello streaming. Il servizio `streaming_service`, basato sulla libreria **Quix**, ascolta Redpanda.
Ma non si limita a leggere. Deve "ricordare".
Per calcolare la media mobile degli ultimi 10 minuti su dati che arrivano uno alla volta, serve una memoria (**State**).
Quix gestisce questo stato in modo trasparente usando **RocksDB** (un database veloce su file locale).
1.  Arriva un messaggio: "Vibrazione 50".
2.  Quix recupera lo stato precedente da RocksDB ("La somma delle vibrazioni precedenti era 450 per 9 messaggi").
3.  Aggiorna lo stato ("Nuova somma 500 per 10 messaggi -> Media 50").
4.  Salva il nuovo stato su RocksDB.
5.  **Data Push**: Una volta calcolata la feature aggiornata, Quix la spinge direttamente dentro **Redis** (tramite Feast), sovrascrivendo il valore vecchio.
In questo modo, Redis è sempre mantenuto "fresco" dal flusso streaming, con una latenza inferiore al secondo.

---

# ATTO SESTO: IL GIUDIZIO (Inferenza & API)

Tutto questo lavoro serve a un solo scopo: rispondere alla domanda dell'utente.
"La macchina 1 ha un problema?"

### 6.1 La Richiesta (FastAPI & Pydantic)
L'utente (o una dashboard) invia una richiesta HTTP POST all'API `inference_service`.
L'API, scritta in **FastAPI**, riceve la richiesta.
Qui entra in gioco **Pydantic**.
*   **Cos'è Pydantic?**: È una libreria di validazione dati. Immaginala come un "buttafuori" all'ingresso del club.
*   Controlla che i dati in arrivo rispettino lo schema definito (es. "L'ID della macchina DEVE essere un numero intero", "Non accetto stringhe vuote").
*   Se l'input è sbagliato, Pydantic lo blocca subito e rispedisce un errore chiaro al mittente, senza far perdere tempo al resto del codice.

### 6.2 Il Recupero (Fetch Online Features)
L'API non riceve i dati dei sensori dall'utente. Riceve solo l'ID (`machine_id=1`).
L'API si gira verso Feast e chiede: "Dammi le features attuali per la macchina 1".
Feast interroga Redis.
Redis risponde istantaneamente con il vettore calcolato poco fa da Quix.
Questo disaccoppia il client (che non deve sapere nulla dei sensori) dal backend (che ha tutti i dati caldi pronti).

### 6.3 La Predizione (Model Inference)
L'API ha il vettore di features. Ora serve il cervello.
All'avvio (`startup`), l'API ha scaricato il modello `AnomalyForest` direttamente dal Model Registry di MLflow. Lo tiene caricato in RAM.
Passa il vettore al modello.
Il modello calcola l'Anomaly Score (-1 anomalia, 1 normale).

### 6.4 La Risposta
L'API impacchetta il risultato in un JSON (`{"is_anomaly": true, "confidence": 0.95}`) e lo restituisce all'utente.
Tutto questo processo (Request -> Redis -> Model -> Response) avviene in meno di 50 millisecondi.

---

# ATTO SETTIMO: LA RESILIENZA (Cosa succede se...)

In un sistema distribuito, le cose si rompono. Ecco come il sistema reagisce ai guasti:

1.  **Se Redis Cade**:
    *   L'API di Inferenza non può più leggere le feature.
    *   Restituirà un errore `503 Service Unavailable`.
    *   Tuttavia, Redpanda continuerà ad accumulare messaggi. Appena Redis torna su, Quix riprenderà ad aggiornarlo. Nessun dato perso, solo un disservizio temporaneo.

2.  **Se MLflow Cade**:
    *   Nessun problema per l'Inferenza. L'API scarica il modello *solo all'avvio*. Una volta partito, il modello è in RAM. L'API continua a funzionare anche se MLflow muore.
    *   Non possiamo però addestrare nuovi modelli o tracciare esperimenti finché non torna su.

3.  **Se Quix (Streaming) si Ferma**:
    *   Redis smette di aggiornarsi. Le feature diventano "vecchie" (Stale Data).
    *   L'API continuerà a rispondere, ma basandosi su dati statici vecchi.
    *   Redpanda accumula i messaggi non letti. Quando Quix riparte, leggerà velocemente tutto l'arretrato ("Burst") e aggiornerà Redis in pochi secondi.

---

# GLOSSARIO TECNICO

*   **Materializzazione**: Il processo di copia dei dati dall'Offline Store (lento, storico) all'Online Store (veloce, recente).
*   **Cold Start**: La situazione in cui il sistema streaming parte da zero e non ha abbastanza storico in memoria per calcolare le finestre temporali. Si risolve caricando lo stato iniziale dal Batch.
*   **Backfill**: L'elaborazione retroattiva di dati storici per riempire buchi o correggere errori nei dati presenti.
*   **Feature Store**: Un sistema che centralizza la gestione delle feature, garantendo che Training (Offline) e Inferenza (Online) usino gli stessi dati e le stesse definizioni.
*   **Anomaly Score**: Un numero (solitamente tra -1 e 1) che indica quanto un dato è "diverso" dalla normalità appresa. Più è basso/negativo, più è anomalo.
*   **Backend Store (MLflow)**: Il database (spesso SQLite o Postgres) dove MLflow salva i metadati (nomi esperimenti, parametri, metriche).
*   **Artifact Store (MLflow)**: Lo storage (Filesystem o S3) dove MLflow salva i file pesanti (modelli .pkl, immagini, dataset).

---

# CONCLUSIONE

Abbiamo trasformato un problema confusionario ("capire se si rompe") in una **pipeline deterministica**.
*   **PySpark** ci ha dato i dati.
*   **Parquet** li ha conservati efficientemente.
*   **Spark & Joblib** hanno creato la conoscenza (features & scaler).
*   **IsoForest & MLflow** hanno creato l'intelligenza (modello).
*   **Feast & Redis** hanno creato la memoria veloce.
*   **Redpanda & Quix** hanno creato il sistema nervoso in tempo reale.
*   **FastAPI** ha creato la voce che parla all'utente.

Ogni componente è stato scelto non perché "famoso", ma perché risolveva uno specifico collo di bottiglia del flusso precedente. Questa è l'essenza di un'architettura MLOps.
