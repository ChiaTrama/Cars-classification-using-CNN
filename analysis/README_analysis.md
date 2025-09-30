# Analisi dei Risultati - Project Cars

Questa cartella contiene gli script e le istruzioni per **analizzare i risultati** degli esperimenti di training e validazione dei modelli.

---

## Cosa sono i summary statistics?

Durante il training, per ogni esperimento viene salvato un file `summary_*.pkl` nella cartella `runs/<nome_strategia>/`.  
Questi file contengono tutte le **metriche principali** raccolte epoca per epoca, oltre a informazioni di configurazione e performance del modello.

### Ogni summary contiene:
- **Configurazione della strategia** (nome, dimensione immagini, augmentations, ecc.)
- **Nome e parametri del modello**
- **Numero di classi**
- **Metriche per ogni epoca**:
  - `train_loss`: lista delle loss di training per epoca
  - `val_loss`: lista delle loss di validazione per epoca
  - `train_acc`, `val_acc`: accuracy top-1 su training/validazione
  - `val_bal_acc`: balanced accuracy su validazione
  - `val_top5`: top-5 accuracy su validazione (se calcolata)
  - `val_f1`: macro F1-score su validazione (se calcolata)
  - `val_roc_auc`, `val_precision`, `val_recall`: metriche aggiuntive per verification
- **Altri dettagli**: epoca migliore, tempo di training, path del checkpoint, batch size, learning rate, ecc.

---

## Come usare i summary

1. **Visualizzazione rapida**  
   Usa lo script `analysis/print_summary.py` per stampare a schermo tutte le metriche salvate in un file summary:
   ```bash
   python analysis/print_summary.py
   ```
   Modifica il path del file summary all'inizio dello script per puntare al file che vuoi analizzare.

2. **Plot delle metriche**  
   Usa lo script `analysis/plot.py` per generare automaticamente i grafici di tutte le metriche disponibili nei summary:
   ```bash
   python analysis/plot.py
   ```
   I plot verranno salvati nella cartella `analysis/plots/`.

3. **Analisi avanzata**  
   Puoi caricare i file summary in qualsiasi notebook Python con:
   ```python
   import pickle
   with open("runs/<nome_strategia>/summary_*.pkl", "rb") as f:
       summary = pickle.load(f)
   ```

---

## Note

- Le metriche sono salvate come **liste** (una voce per ogni epoca) oppure come valori singoli (ad esempio, la migliore accuracy raggiunta).
- Se una metrica appare come valore singolo, significa che è stata calcolata solo sull’ultima epoca.
- I summary sono pensati per essere facilmente estendibili: puoi aggiungere nuove metriche o informazioni senza cambiare la struttura base.

---

## Esempio di struttura di un summary

```python
{
    "train_loss": [...],
    "val_loss": [...],
    "val_acc": [...],
    "val_bal_acc": [...],
    "val_top5": [...],
    "val_f1": [...],
    "val_roc_auc": [...],
    "val_precision": [...],
    "val_recall": [...],
    "best_val_acc": ...,
    "best_epoch": ...,
    "training_time": ...,
    ...
}
```

---

## Consigli

- Per confrontare più strategie/modelli, usa i plot automatici di `plot.py`.
- Per analisi personalizzate, carica i summary in un notebook e lavora direttamente sulle liste delle metriche.
- Se aggiungi nuove metriche, assicurati di salvarle come liste (una voce per epoca) per poterle plottare facilmente.

---