# Screenshot

Catture dell'interfaccia, riprodotte da una sessione reale registrata
(`csi_training_20260924_123332.csv`, ~9.9 pacchetti/s) con la baseline
calibrata sulla fase di stanza vuota.

| File | Cosa mostra |
|---|---|
| `01-monitor-stanza-vuota.png` | Stanza vuota riconosciuta al 97% di confidenza. Il delta CSI resta basso e piatto. |
| `02-monitor-movimento.png` | Persona che cammina fra TX e RX. È il caso più solido: sui dati di test il movimento viene riconosciuto correttamente nel 97–100% dei casi. |
| `03-monitor-presenza-statica.png` | Persona ferma in un punto coperto dal training. È la distinzione più difficile, perché richiede il confronto con la firma della stanza vuota. |
| `04-mappa-radio-fresnel.png` | Campo di sensibilità del link: il lobo segue la prima zona di Fresnel (linea bianca) fra TX e RX, e la sua intensità è l'attività misurata. Non è una localizzazione: con una sola coppia TX–RX non c'è informazione sufficiente per dire *dove* sia la persona. |
| `05-dati-heatmap.png` | Ampiezza normalizzata per sottoportante nel tempo. La struttura in frequenza cambia visibilmente quando lo stato della stanza cambia. |
| `07-risultati-training.png` | Il riepilogo mostrato al termine dell'addestramento. Il primo numero è l'accuratezza sui dati di training ed è ~100% per costruzione: quello da leggere è la CV a blocchi con embargo. La matrice di confusione è calcolata fuori campione, quindi coerente con quella percentuale. |
| `06-fase-tof.png` | Analisi di fase sul blocco LLTF. Il ritardo stimato è dominato dal carrier frequency offset dell'ESP32, quindi vale come indicatore di agitazione del canale, non come tempo di volo. |

Le immagini sono catture della sola finestra dell'applicazione, a 1600×1000.
La barra di stato riporta `REPLAY` perché i pacchetti provengono dal CSV
registrato e non da un ESP32 collegato in quel momento.
