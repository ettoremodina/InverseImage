# PLAN — Revisione della pipeline

Documento di lavoro. **Nessuna implementazione parte finché ogni punto non è marcato
`OK`.** Si lancia tutto insieme a revisione conclusa.

Legenda stato:

| Stato | Significato |
| --- | --- |
| `OK` | approvato, pronto da implementare |
| `RIVEDERE` | direzione accettata, dettaglio ancora da fissare |
| `DOMANDA` | serve una risposta prima di poter decidere |
| `SCARTATO` | valutato e respinto (tenuto a memoria del perché) |

Vincolo trasversale accettato: **codice ordinato e modulare** — ogni aggiunta che ha
senso astrarre ha il suo script, raggruppato in cartelle parlanti. Proposta di layout
in fondo al documento.

Principio di lavoro emerso in revisione: **quando una scelta è di gusto, non si decide a
tavolino — si implementano entrambe dietro un flag e si confrontano sul risultato.**

---

## Stato di implementazione

Fasi 2 e 3 implementate (l'ordine è quello previsto in fondo al documento: lo sciame
è l'ultimo pezzo e beneficia di tutto il resto).

| Punto | Stato | Dove |
| --- | --- | --- |
| (a) celle come cellule | **fatto** | [rendering/cells.py](rendering/cells.py), `CellRenderConfig` |
| (b) supersampling | **fatto** | [rendering/supersample.py](rendering/supersample.py), `render_supersample` |
| (c) luce | **fatto** | [rendering/lighting.py](rendering/lighting.py), `LightingConfig` |
| (d) sfondo scuro, unificato | **fatto** | `PipelineConfig.apply_palette` — un solo valore per entrambi i renderer |
| (e) palette dall'immagine | **fatto** | [color/palette.py](color/palette.py), entrambe le regole di sfondo |
| (f) qualità della linea | **fatto** | `branch_width_mode`, `branch_smoothing`, potatura; pesi Murray in [rendering/tree_weights.py](rendering/tree_weights.py) |
| (g) grading condiviso | **fatto** | [color/grading.py](color/grading.py), un solo passaggio per ogni frame di ogni stadio |
| (h) temporal_smoothing | **fatto** | 0.25 |
| 3.1 timeline unica | **fatto** | [rendering/timeline.py](rendering/timeline.py), `--mode timeline` |
| 3.2 semina progressiva | **fatto (passo 1)** | [nca/seeding.py](nca/seeding.py) + hook in `CAModel.forward`. Il passo 2 (retraining) si decide **guardando il render** |
| 3.3 sovrapposizione NCA↔sciame | **fatto** | [swarm/stage.py](swarm/stage.py) + slot `swarm` in `timeline.render`; il `nutrient` è l'alpha NCA del frame corrente, quindi lo sciame vive solo dove c'è tessuto |
| 3.4 impalcatura che sfuma | **fatto** | `ScaffoldFadeConfig`, entrambe le modalità (`alpha` di default, `time` per confronto) |
| 3.5 camera | **fatto** | [rendering/camera.py](rendering/camera.py), disattivabile |
| 3.6 timing parametrico | **fatto** | [config/timing_config.py](config/timing_config.py), con `validate()` che protegge il vincolo di sovrapposizione |
| Fase 1 — sciame, motore | **fatto** | [swarm/](swarm/) al completo, `particles/` rimosso |
| Fase 1 — sciame, taratura | **in corso, automatizzata** | i parametri non si tarano a mano (§13, "tuning permaloso"): criteri di successo, metriche di immagine e ricerca automatica in [Swarm_Tuning.md](docs/Swarm_Tuning.md) |

Due note di merito emerse implementando, non previste dal piano:

1. **I pesi di Murray sono troppo schiacciati per essere usati grezzi.** Un albero con
   qualche migliaio di foglie ha peso di tronco `sqrt(foglie)`, quindi il 90% dei rami
   finisce a spessore di punta. Serve un `branch_width_gamma` che ridistribuisca —
   è un rimappaggio visivo, la legge resta quella.
2. **La camera non ha bisogno di una canvas più grande.** Ritaglia dentro la canvas
   supersampled che i renderer già disegnano, e il ritaglio si risolve nella stessa
   riduzione ad area che fa l'antialiasing: un solo ricampionamento, mai due.

---

## FASE 1 — Il terzo stadio: sciame evolutivo

### Verdetto sulla diagnosi

`OK` — I tre difetti dello stadio particelle restano il metro di giudizio:

1. il campo di flusso è un Sobel di OpenCV, non viene da nessun modello;
2. **campiona i colori dal target a inference time** — guarda la soluzione;
3. non cresce più niente: NCA congelato, sfondo statico.

### La regola, riformulata

Prima formulazione: *"il target non si guarda a inference"*.

La tua idea della **pressione evolutiva** che punisce chi devia usa il target proprio a
inference — come **pressione selettiva**. Formalmente viola la regola di sopra,
sostanzialmente no, e la
distinzione va messa a verbale perché è il cuore concettuale dello stadio:

> **Il target non può mai fornire un colore. Può solo giudicare.**
> Copiare è vietato, essere giudicati no.

È la stessa relazione che l'NCA ha con il target: la MSE non dipinge nulla, decide solo
quanto vale la regola appresa. La differenza è *quando* accade — e questa differenza è
esattamente ciò che rende lo stadio 3 interessante:

- l'NCA impara **prima**, e a inference esegue;
- lo sciame impara **durante**. L'apprendimento *è* l'animazione.

Chi guarda non vede il risultato di un addestramento: vede l'addestramento. È la
risposta più forte possibile all'accusa iniziale di "gimmick".

`OK` — Regola di progetto confermata. Sostituisce la precedente; tutta la Fase 1 poggia
su questa.

> **Il design completo dello stadio 3 è in [Evolutionary_Swarm.md](docs/Evolutionary_Swarm.md).**
> Qui sotto resta il riassunto operativo; il perché delle scelte sta lì.

Variante severa disponibile in ogni momento (la registro perché è a costo quasi zero):
la fitness calcolata sull'**output dell'NCA** invece che sul target. Lo stadio 3 non
tocca mai il ground truth e diventa "uno sciame che porta ad alta risoluzione l'intento
dell'NCA". Converge però verso un'immagine sfocata e a blocchi, quindi il tetto di
qualità è più basso. Da tenere come flag, non come default.

### L'algoritmo: sciame stigmergico + selezione

Fusione della tua idea con il candidato A (Physarum). Le due metà si completano:
Physarum da solo produce tracce coerenti ma non ha motivo di convergere all'immagine; la
selezione da sola converge ma con movimento incoerente e rumoroso. Insieme: **tratti
coerenti che convergono**.

#### Stato

Per agente: posizione, direzione, **gene di colore**, energia, età.

Campi alla risoluzione di lavoro:

- `canvas` — ciò che è stato dipinto finora;
- `pheromone` — deposito degli agenti, diffonde ed evapora;
- `nutrient` — densità/alpha dell'NCA: il tessuto su cui lo sciame può vivere;
- `error` — scarto tra canvas e target, ricalcolato ad ogni step.

#### Ciclo per step

1. **Percezione** — tre sensori davanti (sinistra / centro / destra) che leggono
   `feromone + w · nutriente`. **Non leggono l'errore**: gli agenti non sanno dove
   stanno sbagliando, lo scopre la selezione. Vedi il parametro `guidance` più sotto.
2. **Movimento** — rotazione verso il sensore più forte, passo avanti.
3. **Deposito** — il colore del proprio gene sul canvas con alpha bassa, più feromone.
4. **Metabolismo** — `Δenergia = guadagno · (errore_prima − errore_dopo) − costo`,
   misurato **solo sui pixel toccati**. Chi migliora l'immagine mangia; chi la peggiora
   dimagrisce; chi ridipinge una zona già corretta guadagna ~0 e muore di fame lo stesso.
5. **Morte** a energia esaurita, **riproduzione** oltre una soglia: il figlio eredita
   posizione, direzione e gene, con **mutazione** sul colore.
6. Popolazione a tetto fisso: i posti liberati dai morti vengono ripresi dai figli dei
   sopravvissuti — i geni si ereditano, non si risorteggiano.

#### Da dove viene il colore, se non dal target

Dal **gene**, ereditato e mutato. Nessun agente legge mai un colore dall'immagine: se si
trova in una regione dove il suo colore funziona sopravvive e si riproduce, e i
discendenti colonizzano quella regione. L'immagine emerge dalla selezione, non da una
lookup.

Popolazione iniziale: geni campionati dall'**output dell'NCA** nel punto di nascita —
legittimo, è il prodotto dello stadio precedente, non il ground truth. Conseguenza
importante: la generazione 0 assomiglia già allo stadio 2, e l'evoluzione **migliora**
il punto di partenza invece di degradarlo. È esattamente la malattia dello stadio
attuale, che parte da un frame buono e lo peggiora.

#### Due livelli di selezione

La tua frase *"ogni pixel o cellula o sezione avrà uno span vitale"* si può prendere alla
lettera su due piani, e funzionano entrambi:

- **agenti** — chi vive e chi si riproduce (sopra);
- **depositi** — anche la pittura ha una vita: il colore che non corrisponde evapora
  in fretta, quello che corrisponde dura a lungo — ma non si fissa mai del tutto
  (Evolutionary_Swarm §5.9). La pressione evolutiva punisce anche la materia, non solo
  chi la depone.

Il secondo livello ha un effetto collaterale prezioso: **il canvas si autopulisce**.
Sporco, aloni e tratti fuori posto svaniscono da soli — cioè il difetto visivo numero uno
dello stadio attuale sparisce per costruzione, senza doverlo inseguire.

#### Comportamenti attesi

Vale la pena scriverli, perché sono il motivo per cui questo stadio è animabile e non
solo corretto:

- territori di colore che si formano e confinano lungo i bordi cromatici del soggetto;
- lo sciame che **si dirada dove l'immagine è finita** e si addensa dove manca —
  emergente, nessuno lo programma;
- **niente estinzione**: il pigmento non cristallizza mai del tutto, quindi il quadro
  consuma sé stesso lentamente e rigenera cibo. La popolazione si assesta su una squadra
  di manutenzione invece di morire. Il finale torna a essere una scelta di regia, e in
  cambio la sequenza è potenzialmente loopabile
  ([Evolutionary_Swarm.md §7](docs/Evolutionary_Swarm.md)).

#### Parametri di regia (tutti in config)

- `fitness_scale` — **la manopola più importante.** La scala a cui si misura l'errore:
  fine = ritratto fedele; grossolana (errore su immagine sfocata/ridotta) = tratti larghi
  e gestuali, resa impressionista. Governa lo stile, non la qualità.
- `tolerance` — errore sotto soglia vale zero: sotto quel livello nessuno guadagna più
  nulla e l'immagine resta **una ricostruzione stilizzata**, non una fotocopia.
- `guidance` (default 0) — quanto gli agenti annusano anche il gradiente dell'errore.
  A 0 è emergenza pura; alzandolo converge molto prima ma "sa dove sbaglia", e si
  avvicina al barare. Esposto per poterlo assaggiare, non per usarlo.
- `deposit_alpha`, `mutation_sigma`, `sensor_angle`, `sensor_distance`, `evaporation`,
  `population_cap`, `energy_cost`, `reproduction_threshold`.

#### Rischi da tenere d'occhio

1. **Convergenza perfetta = fotocopiatrice costosa.** Se lo sciame arriva a errore zero
   hai costruito un modo lento di copiare il target. `tolerance` e `fitness_scale`
   esistono per impedirlo: il risultato interessante è *come* approssima, non *quanto*.
   È il primo parametro da tarare.
2. **Degenerazione golosa** — agenti fermi che ridipingono lo stesso pixel giusto.
   Coperto dalla fitness a *miglioramento marginale* (il secondo deposito rende zero),
   più un costo fisso per deposito.
3. **Tuning sensibile** — i parametri Physarum sono notoriamente permalosi. Serve un giro
   di calibrazione su frame fermo prima di produrre video.
4. **Costo** — il campo errore si ricalcola ad ogni frame: è una sottrazione a piena
   risoluzione, banale su GPU. Popolazione 10-50k agenti, tutto vettorizzato in torch,
   come già fa `particles/refiner.py`. Ordine di grandezza uguale all'attuale.

### Rivalutazione delle proposte precedenti sotto questa luce

| Proposta | Verdetto | Motivo |
| --- | --- | --- |
| A — Physarum | **assorbita** | Diventa il motore di movimento dello sciame. Da sola non convergeva; con la selezione sì. |
| B — Reaction-Diffusion | `SCARTATO` | Non ha individui da selezionare: lo stato è una concentrazione chimica. Si potrebbero evolvere i parametri locali, ma RD *vuole* i suoi pattern e combatte la convergenza. Incompatibile con l'idea. |
| C — CMA-ES sui parametri | `SCARTATO` | Superata. Metteva l'apprendimento fuori, offline, sugli iperparametri; la tua idea lo mette dentro, online, sugli individui. Molto meglio. Resta semmai un anello esterno per tarare i parametri di regia, non un pezzo dell'opera. |
| D — DLA | `SCARTATO` | Compatibile in teoria (aggregare dove migliora, dissolvere dove no) ma il moto è random walk: rumoroso e poco governabile rispetto agli agenti Physarum. |
| Particelle originali | **riabilitate** | Lo sciame resta. Perde la lookup sul target e il flusso da Sobel, guadagna geni e metabolismo. |

### L'arco del progetto, adesso

Il motivo per cui questa scelta chiude il problema di partenza — tre famiglie davvero
diverse, e tre rapporti diversi con l'apprendimento:

| Stadio | Famiglia | Apprendimento |
| --- | --- | --- |
| SCA | crescita geometrica deterministica | nessuno |
| NCA | automa cellulare differenziabile | discesa del gradiente, **offline**, regola condivisa |
| Sciame | popolazione ad agenti stigmergici | **selezione naturale, online, individuale** |

Stesso obiettivo, tre meccanismi opposti. E il terzo è l'unico in cui l'apprendimento è
visibile a schermo.

---

## FASE 2 — Aspetto, senza alzare la risoluzione

### a) Celle rese come cellule — `RIVEDERE`

Scelta: **opzione 2, disegno per cella (Cairo)**. Vincolo tuo: zero impatto sul training
NCA — rispettato per costruzione, è codice di rendering
([nca_renderer.py:24-60](rendering/nca_renderer.py:24)), il training non lo tocca.

Obiezione tua da risolvere: *"il risultato finale deve essere una superficie, non una
colonia"*.

Proposta: **la cellularità è una funzione del tempo, non una proprietà fissa.** Un solo
parametro, il gap tra celle, interpolato sulla maturità:

- cellula giovane (alpha bassa) → raggio piccolo, gap visibile → si leggono gli individui
  mentre crescono;
- cellula matura → raggio oltre il passo della griglia, le vicine si compenetrano, i gap
  si chiudono → **superficie continua**.

Vedi la colonia durante la crescita e il tessuto alla fine. Con la luce del punto (c) la
micro-irregolarità residua per cella legge come pelle, non come griglia di pallini.

Controllo esplicito: `cellularity: float` (0 = superficie liscia, 1 = colonia marcata) e
`cellularity_curve` per decidere come scende nel tempo. A 0 si torna al comportamento
attuale ma antialiasato.

`OK` — Deciso: a fine crescita la cellularità **non va a zero**, resta un residuo appena
percettibile (`cellularity_floor`, ordine di 0.1). A zero netto la superficie diventa
liscia e morta; il residuo minimo la tiene viva e, insieme alla luce del punto (c), la fa
leggere come pelle.

### b) Supersampling — `OK`

Approvato. Si disegna internamente a 2x e si riduce a 512 con media d'area: antialiasing
vero, rami sottili della SCA che smettono di sfarfallare, bordi delle celle morbidi.
Simulazione NCA ferma a 128, video finale fermo a 512, dimensione file invariata. Costo:
~3-4x sul solo tempo di rasterizzazione.

Esposto come `render_supersample: int = 2`, con `1` = disattivato.

### c) Luce al posto dei pixel — `OK`

Normal map dal gradiente dell'alpha smussata → lambertiano + rim light + bloom morbido
sulle celle più luminose. Operazioni 2D su numpy dopo il rendering delle celle.
Direzione luce, intensità rim e soglia bloom in config.

### d) Sfondo scuro — `OK`

Motivazione tenuta: contrasto e profondità (l'argomento "graffi" è stato ritirato, lo
stadio 3 cambia comunque). Il colore **sta in config e deriva dall'immagine di
riferimento** tramite la palette del punto (e), con override manuale che vince sempre.

Da unificare i due punti dove oggi il fondo è hardcoded separatamente
([render_config.py:12](config/render_config.py:12) e
[render_config.py:34](config/render_config.py:34)).

### e) Palette derivata dall'immagine — `OK`

Estrazione con k-means sui soli pixel non trasparenti del target, k ≈ 5, ordinati per
luminanza. Distribuzione:

- **sfondo** — dalla palette (regola sotto);
- **albero SCA** — tinta dominante desaturata e scurita; il gradiente base→punta si
  costruisce sulla stessa tinta invece che marrone→verde
  ([render_config.py:14-15](config/render_config.py:14));
- **celle NCA** — colori propri dell'NCA, invariati;
- **sciame** — i geni iniziali vengono dall'NCA (vedi Fase 1), la palette serve solo per
  eventuali accenti;
- override manuale sempre prioritario.

Regola sfondo: implemento **entrambe** (più scuro desaturato / complementare della tinta
dominante) dietro `background_rule`, default sul primo. Si sceglie guardando due o tre
immagini, non a tavolino.

### f) Qualità della linea SCA — `OK`, con confronto

Due tecniche, entrambe implementate e disattivabili come richiesto:

1. **Spessore per dimensione del sottoalbero** invece che per profondità
   ([sca_renderer.py:173](rendering/sca_renderer.py:173)). La regola è quella osservata
   da Leonardo e poi formalizzata da Murray: un ramo che si divide conserva
   `w_padre² = Σ w_figli²`. In pratica un ramo è spesso in proporzione a **quanto albero
   sostiene**, non a quanto è lontano dalla radice. Oggi due rami alla stessa profondità
   hanno lo stesso spessore anche se uno regge mezza chioma e l'altro un rametto morto:
   è questo che fa leggere l'albero come uno schema tecnico invece che come una pianta.
2. **Smoothing spline** delle polilinee: la SCA emette un segmento per passo di crescita,
   quindi le curve sono spezzate. Interpolarle (Catmull-Rom) le rende curve vere.

Più potatura/decimazione degli ultimi livelli, che oggi sono uno scarabocchio denso.

Flag: `branch_width_mode: 'depth' | 'subtree'`, `branch_smoothing: bool`, e uno switch
per **generare entrambe le versioni in un colpo** e confrontarle affiancate.

Nota implementativa: il peso del sottoalbero si calcola una volta in fase di export, dove
la gerarchia è ancora disponibile ([exporters.py:177](rendering/exporters.py:177)), e si
salva nel JSON accanto a `depths`. Il renderer resta stupido e veloce.

### g) Pass di grading condiviso — `OK`

Il problema che risolve: oggi ogni stadio esce da un percorso di codice diverso, quindi i
tre pezzi sembrano tre export diversi. Una sola funzione applicata a **ogni** frame di
**ogni** stadio, subito prima della scrittura: vignettatura leggera, grana, curva di tono
coerente con la palette, opzionale aberrazione cromatica appena accennata.

La grana in particolare rompe la griglia dei pixel: la bassa risoluzione viene letta come
grana di pellicola invece che come immagine povera.

Costo trascurabile. Tutti i parametri in config, ognuno disattivabile, e l'intero pass
disattivabile in blocco. Default scelto: **grana animata** (rumore nuovo ad ogni frame),
con `grain_animated: bool` per passare a quella fissa se in compressione h264 risulta
troppo rumorosa.

### h) `temporal_smoothing` — `OK`

Oggi a 0 ([render_config.py:38](config/render_config.py:38)). Portarlo a 0.2-0.3 toglie
lo sfarfallio della fase rumorosa. Da tarare a occhio.

---

## FASE 3 — Render finale

### 3.1 Timeline unica — `OK`

Approvato l'unico loop di frame. Oggi `combined.mp4` + `particles.mp4` sono prodotti
separatamente e concatenati con ffmpeg ([render.py:294](render.py:294)), con sfondo
congelato nella fase 3 ([render.py:280](render.py:280)): al taglio il sway dell'albero si
ferma di colpo.

Nuovo: un solo loop, ogni stadio con la sua curva di attività, valutato ad ogni frame.
Va a sostituire `CombinedRenderer`. È il prerequisito di 3.2-3.6.

### 3.2 Sovrapposizione SCA ↔ NCA — `OK`, con esperimento prima

Approvato l'ordine: **prima l'esperimento a modello invariato, poi si decide sul
retraining.**

**Passo 1 — solo rendering.** La profondità di ogni ramo è già calcolata all'export
([exporters.py:177](rendering/exporters.py:177)) e già usata per far crescere l'albero
([sca_renderer.py:159](rendering/sca_renderer.py:159)). Si mappa
`birth_time = depth(tip) / max_depth` e si inietta ogni seme nello stato NCA al frame
corrispondente invece che tutti a t=0 ([data.py:66-68](nca/data.py:66)). Si guarda il
risultato col modello attuale. Costo: un render.

**Cosa si osserva.** Il modello è addestrato con tutti i semi accesi insieme, quindi la
semina progressiva è fuori distribuzione. I due modi di rompersi da cercare: regioni già
cresciute che si perturbano all'accensione di un seme vicino, e semi tardivi che non
attecchiscono.

**Passo 2 — solo se il passo 1 non regge.** Adattamento del training:

- `training.py` — il seme è costruito una volta sola
  ([training.py:105-115](nca/training.py:105)); va randomizzato per campione del batch
  (sottoinsieme casuale, o accensione scaglionata durante il rollout);
- interazione col pool: gli stati conservati conterrebbero configurazioni parzialmente
  seminate, cioè più diversità — non un problema. La logica "rimpiazza il peggiore col
  seme" ([training.py:56](nca/training.py:56)) continua a funzionare;
- costo per step: **nullo**, è una scrittura nel tensore;
- costo in convergenza: compito strettamente più difficile, stima +10-30% di epoche;
- mitigazione: curriculum (prime epoche come oggi) o metà batch classico e metà
  scaglionato.

**Impatto sul codice** (vale per entrambi i passi):

- nuovo `nca/seeding.py` — `SeedSchedule` che produce `(posizione, step di nascita)` e
  `inject(stato, step)`;
- `nca/model.py` — `forward` esegue un ciclo chiuso
  ([model.py:57-60](nca/model.py:57)): serve un parametro opzionale chiamato tra uno step
  e l'altro. Modifica piccola e retrocompatibile;
- `nca/data.py` — invariato.

### 3.3 Sovrapposizione NCA ↔ sciame — `RIVEDERE`

Principio: lo sciame nasce **localmente** dove il tessuto è già maturo (alpha alta e
stabile), non globalmente a NCA finito; e l'NCA continua a girare durante lo stadio 3
invece di congelarsi.

Con lo sciame evolutivo la sovrapposizione diventa più naturale di quanto fosse con le
particelle: il campo `nutrient` è la densità NCA, quindi **gli agenti possono nascere
solo dove c'è già tessuto**. La sovrapposizione non va programmata, è già nella regola —
lo sciame segue la crescita perché non ha altro da mangiare.

Da chiudere dopo la calibrazione dello stadio 3.

### 3.4 L'impalcatura sparisce — `OK`

Quando la carne copre un ramo, quel ramo sfuma. Due implementazioni possibili: guidata
dall'alpha NCA campionata lungo il ramo (più bella) o dal tempo trascorso dalla nascita
(banale). Coerentemente col principio di lavoro: **implemento entrambe dietro flag**,
default sulla prima.

### 3.5 Camera — `OK`, disattivabile

Canvas leggermente più grande, finestra di crop animata: push-in lento e deriva.
`camera.enabled: bool = True`; a `False` finestra fissa e costo extra azzerato.

Nota: camera e supersampling lavorano entrambi su una canvas più grande del video
finale → **un solo fattore di scala interno condiviso**, non due ingrandimenti in
cascata.

### 3.6 Timing — `OK`, tutto parametrico

Requisito tuo: i tempi li scelgo io, ma devono essere **parametri modificabili**. Oggi
sono 40/60 fisso e lineare ([pipeline.py:44-45](config/pipeline.py:44)), e con le fasi
sovrapposte le percentuali disgiunte non hanno più senso: servono intervalli che si
accavallano.

Struttura proposta — ogni stadio ha inizio, fine e curva di easing, tutti in config:

| Parametro | Default proposto | Significato |
| --- | --- | --- |
| `total_duration` | 25 s | durata complessiva |
| `sca.start` / `sca.end` | 0 → 7 s | crescita dell'albero |
| `sca.easing` | `ease_out` | parte svelto, rallenta |
| `nca.start` / `nca.end` | 3 → 16 s | entra mentre l'albero cresce ancora |
| `nca.easing` | `ease_in_out` | fioritura lenta al centro |
| `swarm.start` / `swarm.end` | 12 → 22 s | entra su tessuto già maturo |
| `scaffold_fade.start/end` | 10 → 18 s | l'albero si dissolve sotto la carne |
| `hold` | 3 s | fermo finale col solo respiro del sway |

Default confermati. Vincolo di coerenza da imporre in config: ogni `start` deve cadere
prima della `end` dello stadio precedente, altrimenti le fasi tornano sequenziali e si
perde tutto il lavoro di 3.1.

I titoli/credits (già in [TODO.md](docs/TODO.md)) si agganciano qui, sui secondi finali.

---

## Struttura del codice proposta

```
color/
  __init__.py
  palette.py           # (e) estrazione palette dall'immagine di riferimento
  grading.py           # (g) vignetta, grana, curva di tono — pass condiviso

rendering/
  cells.py             # (a) celle come cellule, con cellularità nel tempo
  lighting.py          # (c) normal map, lambert, rim, bloom
  supersample.py       # (b) scala interna -> downscale
  camera.py            # (3.5) finestra di crop animata
  timeline.py          # (3.1) scheduler unico delle fasi, sostituisce combined_renderer.py

nca/
  seeding.py           # (3.2) SeedSchedule + inject

swarm/                 # stadio 3 — sostituisce particles/
  agents.py            # stato, percezione, movimento, deposito
  selection.py         # consumo, costi, morte, riproduzione, mutazione
  fields.py            # feromone, nutriente, errore, invecchiamento del pigmento
  simulation.py        # il ciclo, il clima, le metriche
  lab.py               # suite di controllo: live / sweep / compare

config/
  palette_config.py    # (d, e)
  grading_config.py    # (g)
  camera_config.py     # (3.5)
  swarm_config.py      # (Fase 1)
  timing_config.py     # (3.6)
```

`swarm/lab.py` è deliberatamente separato dalla pipeline: lo stadio 3 si tara **isolato**,
su un'immagine di test e un surrogato dell'output NCA, prima di essere collegato agli
altri due stadi ([Evolutionary_Swarm.md §11](docs/Evolutionary_Swarm.md)).

`particles/` viene rimosso solo a sciame funzionante, non prima.

---

## Ordine di esecuzione proposto

Da eseguire **tutto insieme** a revisione conclusa; l'ordine è di dipendenza, non di
consegna.

0. `swarm/` + `swarm/lab.py` **isolato** — Fase 1, motore e suite di controllo su
   immagine di test. Non dipende da nient'altro (usa il surrogato dell'NCA) ed è il pezzo
   con più incognite, quindi conviene che la taratura giri in parallelo al resto.
   L'integrazione nella timeline resta al punto 7.
1. `color/palette.py` + config sfondo — (d, e). Non dipende da niente e cambia subito la
   faccia del progetto.
2. `rendering/supersample.py` + `cells.py` + `lighting.py` — (a, b, c, f, h).
3. `color/grading.py` — (g). Dopo che tutti gli stadi hanno una sola via d'uscita.
4. `rendering/timeline.py` + `config/timing_config.py` — (3.1, 3.6). Prerequisito del
   resto.
5. `nca/seeding.py` + esperimento a modello invariato — (3.2). Decisione sul retraining
   **dopo** l'esperimento.
6. Impalcatura che sfuma + camera — (3.4, 3.5).
7. Integrazione dello sciame nella timeline — (3.3). Il motore arriva già tarato dal
   punto 0; qui cambia solo la sorgente di `nutrient`, che diventa l'alpha dell'NCA al
   frame corrente invece di un campo statico.
8. Titoli e taratura finale dei tempi.

---

## Domande aperte da chiudere prima di partire

Le tre domande di piano erano già chiuse: regola del target confermata, cellularità con
residuo minimo, tempi confermati.

Anche le **cinque scelte di design dello stadio 3** sono chiuse — genoma di solo colore,
clima variabile con costante come caso particolare, nessuna cristallizzazione,
OKLab, `fitness_scale` fedele — più due decisioni di metodo: taratura isolata in
laboratorio e durata simulata indipendente dalla durata vista. Dettaglio e motivazioni in
[Evolutionary_Swarm.md §16](docs/Evolutionary_Swarm.md).

**Niente resta aperto.** Tutto il resto è deciso o coperto da flag.
