# BlindEyes 👁️

Sistema di rilevamento presenza e movimento **senza telecamere**, basato su Wi-Fi CSI (Channel State Information) e machine learning. Due ESP32 formano un link radio: ogni volta che una persona attraversa o si trova nel campo tra i due dispositivi, le proprietà del canale Wi-Fi cambiano — BlindEyes analizza queste variazioni per classificare lo stato della stanza in tempo reale.

---

## Come funziona

Quando un segnale Wi-Fi viaggia da un punto A a un punto B, rimbalza su pareti, oggetti e persone. Ogni riflessione altera l'**ampiezza** e la **fase** del segnale su ciascuna delle **64 sottoportanti OFDM** (subcarrier). Il firmware ESP-CSI espone questi dati grezzi via seriale: BlindEyes li legge, li processa e li passa a un classificatore **Random Forest** per determinare se la stanza è:

- `empty` — stanza vuota
- `motion` — presenza in movimento
- `static` — presenza ferma

---

## Hardware necessario

| Componente | Quantità | Note |
|---|---|---|
| ESP32-C6 o ESP32-C5 | 2 | Consigliati per la qualità RF. Funzionano anche C3/S3/S2 |
| Antenna esterna | 2 | L'antenna PCB integrata è sconsigliata — interferisce con la scheda |
| Cavo USB | 1 | Per collegare il RX al computer (solo un cavo serve) |

I due ESP32 devono essere posizionati a **più di 1 metro di distanza** l'uno dall'altro, idealmente su lati opposti della stanza da monitorare.

---

## Firmware ESP-CSI

Il firmware è open source, fornito da Espressif. Va compilato con **ESP-IDF** e flashato separatamente sui due dispositivi.

### 1. Clona il repository del firmware

```bash
git clone https://github.com/espressif/esp-csi.git
cd esp-csi
```

### 2. Installa ESP-IDF (se non già presente)

Segui la guida ufficiale: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/

### 3. Flasha il firmware TX (trasmettitore)

Il TX invia pacchetti ESP-NOW continuamente. Non è collegato al computer durante il funzionamento normale.

```bash
cd esp-csi/examples/get-started/csi_send
idf.py set-target esp32c6      # o esp32c5 / esp32c3 in base al tuo modello
idf.py flash -b 921600 -p /dev/ttyUSB0 monitor
```

Il MAC address del TX verrà stampato nel monitor seriale — annotalo, ti servirà nelle impostazioni di BlindEyes.

> Su ESP32-C6/C5 il MAC di default è `1a:00:00:00:00:00`. Verificalo dal monitor seriale.

### 4. Flasha il firmware RX (ricevitore)

Il RX riceve i pacchetti del TX, misura il CSI e invia i dati grezzi via USB al computer.

```bash
cd esp-csi/examples/get-started/csi_recv
idf.py set-target esp32c6      # deve corrispondere al modello usato
idf.py flash -b 921600 -p /dev/ttyUSB1
```

Dopo il flash, **lascia il RX collegato via USB** al Mac — è l'unico collegamento necessario durante il funzionamento.

---

## Installazione di BlindEyes

### Dipendenze Python

Richiede **Python 3.11** (consigliato via Homebrew su macOS).

```bash
pip install PyQt6 numpy matplotlib scikit-learn joblib pyserial
```

### Avvio

```bash
cd "Blind eyes.py"
python3.11 dashboard.py
```

Oppure usa direttamente l'app bundle **`Blind Eye.app`** trascinandola in `/Applications`.

---

## Configurazione iniziale

Al primo avvio apri le **Impostazioni** (icona ⚙ in alto a destra) e configura:

| Parametro | Descrizione |
|---|---|
| **Porta seriale** | La porta USB del RX, es. `/dev/cu.usbserial-110` |
| **Baud rate** | `115200` (default, non cambiare) |
| **MAC address TX** | Il MAC del dispositivo trasmettitore, es. `1a:00:00:00:00:00` |
| **Dimensioni stanza** | Larghezza e altezza in metri |
| **Posizione TX / RX** | Coordinate X, Y in metri nella stanza |
| **Soglia confidenza** | Soglia minima per accettare una predizione (default: 0.60) |

---

## Training del modello

Il modello pre-addestrato incluso nel repository è stato ottenuto in un ambiente specifico. Per adattarlo alla tua stanza è necessario un nuovo training (circa 6 minuti).

### Procedura guidata

1. Vai nella tab **DATI & TRAINING**
2. Premi **AVVIA SEQUENZA** — il programma guida attraverso 3 fasi da 120 secondi ciascuna:
   - **Fase 1 — Stanza vuota**: esci dalla stanza per 2 minuti
   - **Fase 2 — Movimento**: cammina avanti e indietro tra TX e RX, variando velocità e traiettoria
   - **Fase 3 — Presenza statica**: spostati in vari punti della stanza, fermandoti ~30 secondi per posizione
3. Al termine premi **SALVA TRAINING CSV** per un backup
4. Premi **ADDESTRA MODELLO** — in pochi secondi viene mostrata l'accuracy

### Accuracy attesa

| Campioni per classe | Totale campioni | Accuracy CV 3-fold |
|---|---|---|
| ~52 | ~160 | 80% |
| ~105 | ~316 | 87% |
| ~316 | ~950 | 92% |
| ~1058 | ~3174 | **94%** |

Con i 120 secondi per fase della sequenza guidata si raccolgono normalmente oltre 1000 campioni per classe, raggiungendo circa il **93–94%** di accuracy.

---

## Struttura del progetto

```
Blind eyes.py/
├── dashboard.py          # Applicazione principale (PyQt6)
├── csi_model.pkl         # Modello Random Forest addestrato
├── csi_scaler.pkl        # StandardScaler per la normalizzazione
├── csi_features.json     # Lista delle 130 feature (64 amp + 64 phase + delta + rssi)
└── settings.json         # Configurazione utente

Blind Eye.app/            # App bundle macOS
```

---

## Pipeline tecnica

```
ESP32 TX  ──[ESP-NOW]──►  ESP32 RX  ──[USB 115200]──►  BlindEyes
                                                            │
                                                  parsing trama CSI
                                                            │
                                              64 ampiezze + 64 fasi
                                              + delta ampiezza + RSSI
                                                     (130 feature)
                                                            │
                                                  StandardScaler
                                                            │
                                              Random Forest (200 alberi)
                                                            │
                                          empty / motion / static
```

---

## Risoluzione problemi comuni

**Nessuna connessione seriale**
Verifica che il RX sia collegato via USB e che la porta in Impostazioni corrisponda a quella mostrata in `ls /dev/cu.usbserial-*`.

**ESP-NOW send error (no memory)**
Il canale Wi-Fi è congestionato. Cambia canale nel firmware o spostati in un ambiente con meno interferenze.

**Accuracy bassa dopo training**
Assicurati che durante la fase "presenza statica" tu cambi effettivamente posizione nella stanza ogni 30 secondi — non basta cambiare postura.

---

## Crediti firmware

Il firmware ESP-CSI è sviluppato da **Espressif Systems**: https://github.com/espressif/esp-csi
