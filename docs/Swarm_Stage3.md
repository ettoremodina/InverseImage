# Stadio 3 — mappa: cosa c'è, dove sta, cosa lanciare

Documento di orientamento. Gli altri due parlano di *perché*
([Evolutionary_Swarm.md](docs/Evolutionary_Swarm.md), il design) e di *come si tara in
automatico* ([Swarm_Tuning.md](docs/Swarm_Tuning.md)). Questo dice **dove mettere le mani
e da dove arrivano i numeri che vedi a schermo**.

---

## 1. Il modello mentale, in cinque righe

Uno sciame di agenti dipinge sopra il tessuto cresciuto dall'NCA. Ogni agente ha un
colore ereditato, si muove, deposita pigmento, e vive se quello che deposita avvicina
l'immagine al target. Chi peggiora muore. Il target non fornisce mai un colore: fa solo
da giudice.

Lo stadio 3 gira in due contesti diversi, ed è la prima cosa da tenere separata perché
quasi tutta la confusione nasce da qui:

| | **Laboratorio** | **Produzione** |
| --- | --- | --- |
| Cosa dipinge sopra | un finto output NCA (il target sfocato) | il vero tessuto NCA |
| Risoluzione | 256px | 512px |
| Chi lo lancia | `python -m swarm.lab` | `python render.py` |
| A cosa serve | provare parametri in fretta | vedere il risultato vero |

Il laboratorio è più veloce ma **più gentile**: il suo finto NCA è solo sfocato, mentre
quello vero sbaglia anche i colori. Un parametro che va bene in laboratorio può rendere
molto meno in produzione. È successo davvero, vedi §5.

---

## 2. I file, e quali puoi ignorare

`swarm/` ha dodici file. Non ti servono tutti. In ordine di quanto è probabile che tu
debba aprirli:

### Quelli che potresti voler leggere

| File | Cos'è |
| --- | --- |
| `config/swarm_config.py` | **tutti i parametri dello sciame**. È qui che si mette mano. |
| `swarm/stage.py` | il collegamento con il render: prende il tessuto NCA e restituisce il livello dipinto |
| `swarm/simulation.py` | il ciclo: 10 righe che chiamano i meccanismi in ordine |

### Il motore — funziona, di solito non si tocca

`agents.py` (percezione, movimento, deposito), `selection.py` (chi mangia, chi muore, chi
si riproduce), `fields.py` (i campi: canvas, feromone, errore), `colorspace.py` (OKLab).

### Gli strumenti di misura e di prova

| File | Cos'è | Aggiunto |
| --- | --- | --- |
| `metrics.py` | misura la **popolazione**: vivi, nascite, energia | prima |
| `quality.py` | misura il **quadro**: quanto è migliorato, quanta area è dipinta | in questa sessione |
| `objective.py` | somma le misure in un voto unico | in questa sessione |
| `experiment.py` | esegue un run e salva gli artefatti | prima |
| `report.py`, `tuning_report.py` | producono PNG e pagine HTML | prima / nuovo |
| `animate.py` | scrive gli mp4 | in questa sessione |
| `lab.py` | il laboratorio: confronta configurazioni **che scegli tu** | prima |
| `tuning.py` | la ricerca automatica dei parametri | in questa sessione |

**Se vuoi ignorare tutta la parte automatica, puoi.** `tuning.py`, `objective.py`,
`tuning_report.py` e `config/tuning_config.py` servono solo al comando
`python -m swarm.tuning`. Non li tocca nessun altro percorso: il render non li importa
mai.

---

## 3. Da dove arrivano i numeri — il caso `population_cap`

Questa è la domanda che hai fatto, ed è il difetto peggiore dell'intera catena.

In `config/swarm_config.py` leggi:

```python
population_cap: int = 10000     # (era 4096)
```

ma il render stampa `16384 agents`. Il motivo è in `SwarmStageConfig`, poche righe più
sotto nello stesso file:

```python
population_scale: bool = True   # scale population_cap with work_size vs lab 256
```

e in `swarm/stage.py`:

```python
if stage_config.population_scale:
    population = int(population * (work_size / 256.0) ** 2)
```

Cioè: il numero che scrivi vale **a 256px**, e in produzione (512px) viene moltiplicato
per (512/256)² = **4**. 4096 × 4 = 16384. Con il tuo 10000 diventerebbe 40000.

L'idea era "mantieni la stessa densità di agenti per pixel". L'effetto pratico è che il
numero in config non è il numero che gira, e te ne accorgi solo leggendo il log.

**Proposta, da approvare tu:** togliere `population_scale`, lasciare che
`population_cap` significhi letteralmente quel numero, e scrivere in config il valore di
produzione. Una riga in meno e un'indirezione in meno. Dimmi se la faccio.

### La catena completa, per capire chi vince

Quando lanci un render, il valore finale di ogni parametro si forma così, **in
quest'ordine** — l'ultimo che tocca un parametro vince:

```
1. config/swarm_config.py        SwarmConfig      i default
2. config/lab_config.py          PRESETS[nome]    il preset scelto in SwarmStageConfig.preset
                                                  (ora 'tuned', che legge config/tuned_swarm.json)
3. swarm/stage.py                                 forza work_size, device, e moltiplica
                                                  population_cap se population_scale=True
4. riga di comando               --swarm-set      solo per quel render, niente viene scritto
```

Il passo 2 è il motivo per cui cambiare `deposit_alpha` in `swarm_config.py` **non ha
effetto** sul render: il preset `tuned` lo sovrascrive. Per cambiarlo davvero, o modifichi
`config/tuned_swarm.json`, o usi `--swarm-set`, o metti `preset = 'rooted'` /
`preset = 'defaults'` in `SwarmStageConfig`.

Se questa catena ti sembra troppo lunga: sono d'accordo. Si può accorciare, ma è una
modifica che voglio farti approvare prima.

---

## 4. I comandi, dal più economico al più caro

### a) Un fermo immagine — ~20 secondi

Il modo per provare i parametri. Rende **solo il fotogramma finale** (secondo 22, a
sciame concluso), non i 500 del video:

```bash
python render.py --mode still --reuse-frames
```

Esce in `outputs/rendering/jellyfish_still_22.0s.png`. Ogni lancio sovrascrive lo stesso
file: rinominalo se vuoi confrontare.

Per cambiare un parametro solo per quel render, senza toccare niente:

```bash
python render.py --mode still --reuse-frames --swarm-set deposit_alpha=0.30
```

Per il termine di paragone senza stadio 3 (esce con `_noswarm` nel nome):

```bash
python render.py --mode still --reuse-frames --no-swarm
```

`--reuse-frames` riusa i frame NCA già su disco (`outputs/rendering/*_nca_frames.npz`)
invece di rieseguire il modello. Senza, aggiungi un paio di minuti.

### b) Il laboratorio — ~30 secondi, ma sul finto NCA

Confronta due configurazioni a parità di seed, o guarda lo sciame muoversi dal vivo:

```bash
python -m swarm.lab --mode live                              # finestra interattiva
python -m swarm.lab --mode preset --preset tuned             # un run, artefatti su disco
python -m swarm.lab --mode compare --preset rooted --preset-b tuned
```

Ricorda: qui il target è sfocato, non è il vero NCA.

### c) Il video intero — ~7 minuti

```bash
python render.py --mode timeline --reuse-frames
```

Sovrascrive `outputs/rendering/jellyfish_timeline.mp4`.

### d) La ricerca automatica — ~50 minuti

```bash
python -m swarm.tuning --space core --workers 6
```

Non lanciarlo se non vuoi aspettare. Scrive tutto sotto
`outputs/swarm/tuning/<data>_<spazio>/`, e `index.html` lì dentro si può aprire mentre
gira. È l'unico comando che ha bisogno di `config/tuning_config.py`.

---

## 5. Dove siamo, in numeri onesti

Contributo dello sciame in produzione, misurato sul tessuto (non su tutto il frame, che
è per l'80% sfondo già corretto):

| configurazione | migliora l'immagine di | dipinge |
| --- | --- | --- |
| preset `tuned`, com'è ora | +1,5% | 7% del soggetto |
| 40 000 agenti invece di 16 000 | +1,8% | 8% |
| a 1024px invece di 512 | +1,7% | 9% |
| `deposit_alpha` 0.30 e `decay_max` 0.08 | **+5,1%** | **15%** |

Da cui le due cose da sapere:

1. **Gli agenti non sono la leva.** Costano quasi nulla su GPU (10 000 e 100 000 danno lo
   stesso tempo, ~15 s per l'intera finestra) ma non cambiano quasi niente nell'immagine.
2. **Le leve sono l'opacità del tratto (`deposit_alpha`) e la velocità con cui il
   pigmento riaffonda nel tessuto (`decay_max`).** Alzando la prima e abbassando la
   seconda si vede. Il tuning automatico non le sceglierebbe mai, perché ha peso 0,24
   sulla fedeltà e pennellate audaci sbagliate la peggiorano.

Il preset `tuned` è stato calibrato in laboratorio a 256px contro il finto NCA, ed è lì
che vale +8,5%. In produzione, contro il vero NCA, vale +1,5%. **Il laboratorio è troppo
gentile**: è il limite più importante di tutto il lavoro di taratura fatto finora.

---

## 6. Se vuoi tornare indietro

Niente di nuovo è obbligatorio.

- `SwarmStageConfig.preset = 'rooted'` torna alla taratura a mano precedente;
  `'defaults'` ai valori del documento di design.
- Cancellando `config/tuned_swarm.json` il preset `tuned` semplicemente sparisce e tutto
  il resto continua a funzionare.
- `render.py --mode timeline --no-swarm` rende i primi due stadi senza il terzo.
- `outputs/rendering/jellyfish_timeline_pre_tuning.mp4` è il video com'era prima.
