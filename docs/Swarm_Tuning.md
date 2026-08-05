# Taratura dello sciame — criteri, metriche, suite automatica

Documento operativo dello stadio 3. Presuppone
[Evolutionary_Swarm.md](docs/Evolutionary_Swarm.md), che dice *cosa* fa lo sciame e
*perché*; qui si stabilisce **quando è tarato bene** e si descrive la macchina che ci
arriva da sola.

Il punto di partenza è una constatazione scomoda, misurata e non opinabile: il preset
`rooted`, che il laboratorio chiamava "la baseline calibrata", **peggiora l'immagine
dello 0,46%** rispetto a ciò che lo stadio 2 gli consegna, ridipinge il 14% del
soggetto e tiene la popolazione inchiodata al pavimento di ripopolamento. Non è un
difetto di implementazione: è che dodici parametri accoppiati non si tarano a mano, e
§13 lo aveva previsto (*"tuning permaloso"*).

Tre pezzi, in ordine:

| | Cos'è | Dove |
| --- | --- | --- |
| §1 | I criteri di successo, presi dal documento di design e resi misurabili | [config/tuning_config.py](config/tuning_config.py) |
| §2 | Le metriche che li misurano | [swarm/quality.py](swarm/quality.py), [swarm/objective.py](swarm/objective.py) |
| §3 | La suite che cerca i parametri finché i criteri non sono soddisfatti | [swarm/tuning.py](swarm/tuning.py) |

---

## 1. I criteri di successo

Non sono nuovi: erano già tutti in `Evolutionary_Swarm.md`, sparsi fra l'arco narrativo
(§7), la diagnostica (§12) e i modi di fallire (§13), scritti però in prosa. Qui
diventano dieci numeri con una banda di accettazione e un peso.

La regola che governa la traduzione: **ogni criterio è una funzione continua, mai un
pass/fail.** Una soglia netta darebbe alla ricerca un muro piatto invece di una
salita — una configurazione che manca di un capello e una che manca di un chilometro
varrebbero identiche zero, e l'ottimizzatore vagherebbe a caso.

### 1.1 Migliora l'immagine? (peso 0,46)

| Criterio | Misura | Banda | Peso |
| --- | --- | --- | --- |
| `improvement` | riduzione dell'errore **sul soggetto** rispetto all'output NCA | 0 → 0,12 | 0,24 |
| `detail_ratio` | energia dell'alta frequenza rispetto al target | 0,85–1,15 | 0,12 |
| `alignment_gain` | quanto meglio i gradienti del canvas si allineano a quelli del target | 0 → 0,05 | 0,10 |

`improvement` è **mascherato al tessuto**, ed è una correzione importante e non
cosmetica: nel target di laboratorio l'82% dei pixel è sfondo, dove base e target già
coincidono. L'errore medio globale è quindi diluito di circa cinque volte, e un
guadagno reale del 2% sulla medusa si legge come 0,4%. Tutte le vecchie percentuali
degli studi vanno lette con questa lente.

Il pieno dei voti a `improvement` è a 12%, non a 100%: §13 chiama *fotocopiatrice* la
convergenza a errore zero, e un criterio che continuasse a premiare oltre spingerebbe
la ricerca esattamente lì.

`detail_ratio` e `alignment_gain` vanno letti insieme, ed è il motivo per cui sono due.
§3 dice che allo sciame resta da mangiare *dettaglio*: l'input è a 128px risalito, la
sua alta frequenza vale 0,385 di quella del target. Aggiungere rumore alza
`detail_ratio` **e abbassa** `alignment_gain`; ricostruire il dettaglio giusto li alza
entrambi. Nessuno dei due, da solo, distingue le due cose.

### 1.2 Sembra dipinto? (peso 0,32)

| Criterio | Misura | Banda | Peso |
| --- | --- | --- | --- |
| `stroke_coherence` | anisotropia del tensore di struttura dello strato di pigmento | 0,25 → 0,55 | 0,12 |
| `coverage` | quota di soggetto effettivamente ridipinta | 0,50–0,98 | 0,08 |
| `chroma_ratio` | croma media rispetto al target | 0,92–1,15 | 0,06 |
| `flicker` | variazione del canvas per step a regime | 0,0002–0,002 | 0,06 |

`stroke_coherence` è l'unico numero del progetto che parla di estetica e non di
ricostruzione, ed è quello che permette di chiedere alla suite **un quadro invece di
una fotocopia**. Misura se lo strato di pigmento varia lungo una direzione sola —
una pennellata — o in tutte allo stesso modo — una macchia, o del rumore per pixel.
Riferimento: lo strato di dettaglio mancante del target stesso legge 0,50.

`flicker` è la lettura diretta del compromesso di §5.9. A zero il quadro ha
cristallizzato, che §16 vieta; troppo alto è l'*equilibrio rumoroso* di §13, un quadro
che non si definisce mai. La banda corrisponde, a 12 step per frame in produzione, a un
movimento percettibile ma non a uno sfarfallio.

### 1.3 È davvero una popolazione che evolve? (peso 0,22)

| Criterio | Misura | Banda | Peso |
| --- | --- | --- | --- |
| `population_fill` | popolazione a regime come frazione del tetto | 0,15–0,90 | 0,08 |
| `birth_rate` | riproduzioni per agente per step | 0,0002 → 0,004 | 0,07 |
| `floor_fraction` | quota di run passata al pavimento di popolazione | 0,5 → 0 | 0,07 |

Sono i tre modi in cui lo sciame può produrre un'immagine accettabile **smettendo di
essere quello che il progetto dice di essere**. Popolazione al pavimento: vive di
ripopolamento, non di cibo, e la selezione non agisce mai (§13, *estinzione precoce*).
Popolazione al tetto con zero morti: nessuno viene selezionato. Zero nascite: niente
eredità, niente mutazione, niente evoluzione — resta uno spruzzatore casuale, che è
precisamente l'accusa di *gimmick* a cui tutto il documento di design risponde.

### 1.4 La regola non negoziabile

Un run che restituisce un'immagine **peggiore** di quella ricevuta dallo stadio 2 è un
fallimento, qualunque cosa abbia ottenuto sul resto: texture bella su immagine
degradata è comunque immagine degradata. Il punteggio composito viene quindi
moltiplicato per un fattore che dimezza a −2% di danno e continua a scendere.

Curva e non interruttore, per una ragione trovata sul campo: le prime configurazioni
stanno a −13%, e una rampa che le azzerasse tutte darebbe alla ricerca zero informazione
proprio nella regione da cui deve uscire.

### 1.5 Cosa **non** è un criterio

`guidance` non compare in nessuno spazio di ricerca. È la manopola del barare (§2.2) e
un ottimizzatore a cui la si desse la troverebbe subito, perché è il modo più rapido in
assoluto di abbassare l'errore. Il preset `guided` resta come **soffitto** contro cui
leggere i run onesti, mai come impostazione di output.

---

## 2. Le metriche

Due file, con una divisione netta di competenze.

**[swarm/metrics.py](swarm/metrics.py) guarda la popolazione** — errore, vivi, nascite,
energia, diversità genetica. È la diagnostica di §12 e basta a distinguere un run che
muore di fame da uno sano. Non basta a distinguere un run che ha ridipinto il soggetto
da uno che l'ha spalmato, perché quella è una proprietà dell'immagine.

**[swarm/quality.py](swarm/quality.py) guarda il quadro.** Misura fedeltà (mascherata al
tessuto), dettaglio, pennellata e colore sul canvas finale. Non è letto da nessun
meccanismo della simulazione: è misura del giudice, applicata a cose fatte.

Una metrica nuova sta invece dentro il ciclo, perché è temporale e non si può ricostruire
dopo: `StepMetrics.canvas_delta`, la variazione media del canvas per step sul tessuto.
Costa un clone e una sottrazione per **riga registrata** (non per step), quindi
`metrics_stride` continua a governarne il costo.

**[swarm/objective.py](swarm/objective.py)** fa solo aritmetica: unisce i due record,
calcola tre letture derivate (`alignment_gain`, `population_fill`, `birth_rate`),
applica i criteri e restituisce il totale **insieme alla scomposizione**. Ogni report
stampa la scomposizione accanto al punteggio: senza, la configurazione vincente sarebbe
solo un altro numero magico, e con, si legge "la fedeltà c'è, è che non si assesta mai".

I due record si sovrappongono su tre nomi — entrambi misurano errore e miglioramento —
e non vogliono dire la stessa cosa. Vince il nome semplice la versione mascherata,
perché è quella che risponde alla domanda; la globale resta con prefisso `global_`
invece di sparire, perché **uno scarto grande fra le due significa che lo sciame sta
dipingendo lo sfondo**.

---

## 3. La suite di tuning

```bash
python -m swarm.tuning                                   # la ricerca di default
python -m swarm.tuning --space full --generations 60
python -m swarm.tuning --space economy --workers 8
python -m swarm.tuning --adopt                           # installa il vincitore
```

### 3.1 L'algoritmo

Una strategia evolutiva (μ+λ) sul cubo unitario normalizzato, con un passo adattivo:

1. `initial_random` sonde iniziali, di cui una è il preset `seed_from` — la ricerca non
   parte mai meno informata dell'ultima taratura a mano. Le altre sono stratificate e
   non uniformi: con trenta sonde in dodici dimensioni, il caso puro lascia intatti
   interi decenni degli assi logaritmici.
2. Ogni generazione, λ figli attorno ai μ eliti: genitore scelto per rango, passo
   gaussiano, **riflesso** ai bordi del cubo e non troncato — troncare rende i bordi
   attrattori e fa concludere "l'ottimo è al limite" quando non lo è.
3. Genitori e figli competono insieme (il `+` di (μ+λ)): un elite sopravvive finché
   qualcosa non lo batte davvero, che conta quando l'obiettivo è rumoroso.
4. Il passo cresce dopo una generazione che migliora e si accorcia dopo una che no.

La scelta è la cosa più semplice che funziona in questo regime — obiettivo non
differenziabile, valutazioni da secondi, budget da centinaia — non la più alla moda: la
griglia non regge dodici dimensioni e l'ottimizzazione bayesiana vuole più struttura di
quanta ce ne sia. Ha anche una proprietà gradevole: **è lo stesso algoritmo della cosa
che sta tarando, un livello sopra.** Una popolazione, una mutazione, e una selezione che
può solo dire di no.

Gli spazi di ricerca sono nominati in `SPACES`: `economy`, `stroke`, `pigment`,
`perception`, `core` (default, 13 parametri) e `full` (23). Ogni parametro dichiara
scala lineare o logaritmica; il metabolismo è tutto logaritmico, perché fra 1 e 10 c'è
lo stesso tipo di differenza che fra 100 e 1000.

### 3.2 Nessun agente nel giudizio

Il criterio è aritmetica e le immagini sono per la persona che legge il report dopo.
Non c'è nessun punto del ciclo in cui un modello linguistico valuta un risultato: la
suite gira per ore da sola e costa zero token.

Il contraccolpo va detto: la suite ottimizza **quello che è scritto in
`config/tuning_config.py`**, non il gusto di chi guarda. Quando il risultato è
"tecnicamente migliore ma non mi piace", la mossa corretta è spostare un peso o una
banda lì dentro e rilanciare — non correggere a mano il vincitore.

### 3.3 Cosa lascia per terra

```
outputs/swarm/tuning/<timestamp>_<spazio>/
  index.html          la pagina da aprire: migliore corrente, curve, filmati
  history.csv         ogni valutazione, ogni metrica, ogni parametro
  progress.png        curve di punteggio, pagella del migliore, deriva dei parametri
  best.json           gli override vincenti — si danno in pasto a `lab.py --config`
  generations/genNN/  canvas, filmstrip, diagnostica e **mp4** del migliore
  progress.mp4        una posa per generazione: la ricerca, come film
  final/              il vincitore rieseguito lungo, su più seed, con video
```

Tutto viene riscritto da capo a ogni generazione, così `index.html` si può aprire
**mentre la ricerca gira** e mostra lo stato dell'ultima generazione conclusa.

Il terzo pannello di `progress.png` merita una riga: è un grafico a coordinate parallele
degli eliti nello spazio normalizzato, e risponde a "la ricerca ha deciso?" parametro per
parametro. Un asse su cui gli eliti stanno tutti sovrapposti ha convertito e si può
congelare; uno su cui sono ancora sparsi o non conta o non è stato risolto — e
distinguere i due casi guardando un CSV è disperante.

### 3.4 I filmati

`--animate-every N` (default 5) gira un film del migliore corrente ogni N generazioni,
su `animation_steps` step invece che sugli step di valutazione: la ricerca valuta corto
per throughput, ma un film di quella lunghezza si fermerebbe prima del regime che
l'arco di §7 deve raggiungere. Il pannello è **input dello stadio 2 · sciame · target**,
che è l'unica inquadratura in cui "ha aiutato?" si risponde a occhio.

`progress.mp4` filma invece la ricerca: una posa tenuta per generazione. Risponde a una
domanda che la curva del punteggio non pone — **cosa ha barattato l'ottimizzatore**
mentre il numero saliva.

### 3.5 Dal laboratorio alla produzione

`--adopt` scrive il vincitore in `config/tuned_swarm.json`, che
[config/lab_config.py](config/lab_config.py) carica all'import come preset `tuned`.
Da lì la produzione lo usa per nome:

```python
SwarmStageConfig.preset = 'tuned'
```

JSON e non numeri incollati nel sorgente perché la taratura resti **un risultato** —
qualcosa che un run ha prodotto e sa riprodurre — invece di una costante digitata una
volta e poi scollegata dallo studio che la giustificava. Senza il file, il preset `tuned`
semplicemente non esiste e tutto il resto si comporta come prima.

---

## 4. Riferimenti misurati

Tutti su `images/jellyfish.png`, 256px, surrogato a 128px, 4096 agenti.

| lettura | input stadio 2 | copia perfetta del target | preset `rooted`, 600 step |
| --- | --- | --- | --- |
| `improvement` (mascherato) | 0,000 | 1,000 | **−0,005** |
| `detail_ratio` | 0,385 | 1,000 | 0,386 |
| `gradient_alignment` | 0,875 | 1,000 | 0,875 |
| `stroke_coherence` | 0,000 | 0,496 | 0,276 |
| `coverage` | 0,000 | 0,945 | 0,141 |
| `chroma_ratio` | 0,972 | 1,000 | 0,972 |

La colonna centrale non è un obiettivo — è la fotocopiatrice di §13 — ma è la scala
naturale: dice quanto leggono questi numeri quando l'immagine è *giusta*, e permette di
mettere una banda fra "non ha fatto niente" e "ha barato" invece che a una costante
arbitraria.

La colonna di destra è il motivo per cui questo documento esiste.
