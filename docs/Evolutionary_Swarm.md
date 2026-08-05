# Evolutionary Swarm — design dello stadio 3

Documento di progetto. **Nessuna implementazione.** Serve a fissare l'idea prima di
scrivere codice, e a lasciare traccia del *perché* di ogni scelta.

Stadio 3 della pipeline, sostituisce `particles/`. Riferimento operativo:
[PLAN.md](docs/PLAN.md).

Le cinque domande aperte sono chiuse — le decisioni stanno in §16. Le sezioni §5-§7
sono state riscritte di conseguenza: sono la specifica meccanica vera e propria.

---

## 1. Il principio

Uno sciame di agenti dipinge sopra il tessuto cresciuto dall'NCA. Ogni agente porta un
**gene di colore**, si muove secondo regole proprie, deposita pigmento e **vive finché
quello che deposita migliora l'immagine**. Chi peggiora dimagrisce e muore. Chi migliora
mangia e si riproduce, e il figlio eredita il gene con una mutazione.

Il target non viene mai letto come colore. Interviene solo come giudizio:

> **La pressione evolutiva ha potere di veto, non potere creativo.**

La distinzione non è retorica, è informazione. Tutto il colore che finisce sul quadro è
passato attraverso il canale genetico — mutazione propone, selezione dispone. Il target
può solo dire *no*: può cancellare, mai creare. Nel limite, la selezione può scegliere
soltanto fra ciò che la mutazione ha proposto. Un algoritmo che campiona
`target[y, x]` — come fa oggi
[refiner.py:262](particles/refiner.py:262) — **copia**; questo **cerca**.

Ed è la ragione per cui lo stadio 3 smette di essere un gimmick: l'NCA impara prima e a
inference esegue, lo sciame impara **durante**. L'animazione non è il risultato di un
addestramento, è l'addestramento.

---

## 2. Perché converge, se nessuno sa dove sta sbagliando

È il punto meno ovvio del progetto, quindi va sviscerato prima di scrivere una riga.

Gli agenti **non percepiscono il campo di errore**. I sensori leggono feromone e
nutriente, nient'altro. Nessuno sa dove sono gli sbagli. Eppure il sistema converge, e
converge per tre meccanismi che si compongono:

**a) L'ereditarietà della posizione trasforma la ricerca casuale in ricerca locale.**
Il figlio nasce dove sta il genitore. Una stirpe che ha trovato una regione dove il suo
colore funziona **resta lì**. Senza questo, ogni agente sarebbe un tentativo casuale
indipendente e non convergerebbe mai in tempo utile.

**b) La mutazione sul colore è una risalita del gradiente senza gradiente.**
Il figlio esplora un colore vicino a quello del genitore. In ogni regione si forma di
fatto una piccola strategia evolutiva `(1+λ)` che ottimizza localmente il colore, senza
che nessuno calcoli una derivata. Il gradiente non serve: lo sostituisce la morte.

**c) Il feromone è un segnale di reclutamento.**
Chi mangia deposita più feromone e attira altri agenti. Attira anche quelli col colore
sbagliato, che moriranno — ma è proprio quello che serve a una regione affamata:
**varietà genetica su cui selezionare**. Il feromone non porta la soluzione, porta
candidati.

I tre insieme fanno una **ricerca locale distribuita**: la posizione dice *dove*, la
mutazione dice *cosa provare*, il feromone dice *dove serve gente*. Nessuno dei tre
conosce il target.

### 2.1 Cosa evolve, cosa no, e perché non serve un cervello

Va detto senza ambiguità, perché è la cosa che si fraintende più facilmente guardando il
ciclo di §5: **gli agenti non hanno un cervello.** Non c'è policy, non c'è rete, non c'è
nessuna regola di decisione che vari da individuo a individuo. La regola di moto è
cablata e identica per tutti, e **un carattere che non varia non può essere selezionato**:
nella versione base il comportamento è una costante di specie, non un tratto individuale.

Quello che evolve è due cose, e sono di natura diversa.

**Il colore evolve in senso pieno.** Eredità, mutazione, selezione differenziale. In una
regione dove il target è un arancio `(L .72, a .08, b .12)` e l'NCA ha lasciato uno
slavato `(.70, .04, .06)`:

| | gene | esito |
| --- | --- | --- |
| capostipite | `(.70, .04, .06)` | depone il colore già presente → guadagno ≈ 0 → dimagrisce |
| figlio | `(.71, .06, .09)` | mutazione fortunata: avvicina il canvas → mangia → si riproduce |
| altro figlio | `(.73, .12, .18)` | ha esagerato: peggiora → perde energia → muore |
| nipote | `(.72, .07, .11)` | ci siamo |

Nessuna derivata, nessuna lettura di `target[y, x]`. Lo sciame ha **misurato** il colore
locale del target per tentativi e morte. È una `(1+λ)` indipendente per ogni regione
dell'immagine, e converge in una decina o poche decine di generazioni per regione — il
che, a `reproduction_threshold` tipici, spiega perché servano migliaia di step di
simulazione e giustifica la decisione di §8 (simulare lungo, accelerare in riproduzione).

**La posizione si eredita ma non è un'abilità.** Il figlio nasce dove sta il genitore con
un jitter: si trasmette *con mutazione*, esattamente come il colore. Il genoma effettivo
non è di 3 numeri ma di **5** — `(x, y, L, a, b)` — e lo sciame fa una risalita locale in
cinque dimensioni. Conseguenza pratica: `birth_jitter` **è un parametro di ricerca**, non
un dettaglio estetico; è il tasso di mutazione della coordinata spaziale.

Ma il carattere selezionato è *il posto*, non la capacità di trovarlo. Nessun individuo
diventa più bravo a cercare: è la popolazione che finisce nei posti giusti, perché chi sta
nei posti sbagliati muore. È dinamica di popolazione, non intelligenza. E la scoperta di
zone nuove, nella versione base, non è nemmeno evolutiva — è affidata al jitter di
nascita, al feromone e alla fame, che di genetico non hanno niente.

### 2.2 Il limite invalicabile, e l'unica scappatoia

Sognare cervelli più grossi serve a poco, per una ragione strutturale: **i sensori non
vedono l'errore.** Nemmeno una rete neurale per agente potrebbe imparare "vai dove ci
sono gli sbagli", perché quell'informazione non le arriva. Potrebbe imparare al massimo
a evitare la folla o a restare sul tessuto.

Con un'eccezione, ed è l'unica strada per rendere efficiente il *singolo* invece della
popolazione: un agente non può vedere l'errore, ma **può sentire la propria energia**.
Sapere "sto mangiando o no" non gli dice dov'è la risposta, gli dice solo se ciò che ha
appena fatto ha funzionato. La regola che ne discende è §5.2b.

La linea di demarcazione col barare, formulata in modo che regga a ogni estensione futura:

> **Leggere l'errore nello spazio è barare** — vedo dove sbaglio *prima* di andarci: è
> `guidance`.
> **Leggere la propria energia nel tempo è legittimo** — so solo se quello che ho fatto
> è andato bene.

Il secondo è lo stesso canale di veto che usa già la selezione. Cambia soltanto la scala
temporale: l'apprendimento passa da *fra le generazioni* a *dentro la vita del singolo*.

---

## 3. Cosa mangia lo sciame — `DECISO`

Domanda apparentemente tecnica, in realtà decide tutto lo stile. Da cosa parte il canvas?

- **Canvas vuoto** — lo sciame ricostruisce tutto da zero. Drammatico, ma butta via lo
  stadio 2 e per parecchi secondi il video *peggiora*: è precisamente la malattia
  attuale.
- **Canvas = output NCA upsamplato** — nessuna regressione visiva, ma l'errore è già
  basso: poco cibo, dinamica fiacca.

**Deciso: canvas = output NCA upsamplato.** L'obiezione "poco cibo" si risolve da sé
guardando *che tipo* di errore resta:

> Quell'immagine è a 128 px e upsamplata: per costruzione le manca l'alta frequenza.
> **L'errore residuo è quasi tutto dettaglio.** Lo sciame mangia dettaglio.

Lo sciame non rifà l'immagine: aggiunge esattamente ciò che l'NCA non era in grado di
rappresentare. È rifinitura vera — non un filtro applicato sopra, ma informazione nuova
guadagnata per selezione. E i geni della popolazione iniziale si campionano dall'output
NCA nel punto di nascita, quindi la generazione 0 parte già plausibile e l'evoluzione
**migliora** il punto di partenza invece di degradarlo.

Conseguenza di cui tenere conto in §7: l'output NCA è anche lo **strato di fondo** verso
cui il pigmento decade. Non si torna mai al vuoto, si torna sempre al tessuto.

---

## 4. Stato e campi

**Per agente** (tutti tensori `(N,)` o `(N, k)`, `N = population_cap`):

| Nome | Forma | Cos'è |
| --- | --- | --- |
| `pos` | `(N, 2)` float | posizione in pixel della griglia di lavoro |
| `dir` | `(N,)` float | angolo in radianti |
| `gene` | `(N, 3)` float | colore in OKLab |
| `energy` | `(N,)` float | riserva; sotto zero si muore |
| `age` | `(N,)` int | step dalla nascita |
| `alive` | `(N,)` bool | slot occupato |

**Campi alla risoluzione di lavoro `(H, W)`:**

| Campo | Cos'è | Chi lo legge |
| --- | --- | --- |
| `canvas` | ciò che è stato dipinto, in OKLab | il giudizio, il rendering |
| `base` | output NCA upsamplato — lo strato sotto | il decadimento del pigmento (§7) |
| `pheromone` | deposito degli agenti; diffonde ed evapora | i sensori |
| `nutrient` | densità/alpha dell'NCA — il tessuto vivibile | i sensori, la nascita |
| `target` | il ground truth in OKLab | **solo `error`** |
| `error` | `‖canvas − target‖` per pixel, alla scala di fitness | **solo il metabolismo** |

Riga chiave della tabella: `error` non è leggibile dai sensori. È l'unico posto dove il
target entra nel sistema, e ci entra solo per decidere chi mangia.

**Perché tutto in OKLab e non in RGB** (§11): il canvas vive in OKLab dall'inizio alla
fine, e si converte in sRGB solo al momento di scrivere il frame. Così l'alpha-blending
del deposito, la distanza dell'errore e la mutazione dei geni parlano tutti la stessa
lingua percettiva, e non c'è una conversione per agente per step.

---

## 5. Il ciclo, passo per passo

Questa è la sezione operativa: ogni sotto-passo ha la formula, i parametri che lo
governano e le alternative valutate. **★ = scelta raccomandata come default.**

Un vincolo trasversale che vale per tutto il ciclo: **ogni passo deve essere
order-independent**. Tutti gli agenti leggono lo stato *dello step precedente* e
scrivono in accumulatori; nessuno legge quello che un altro ha appena scritto. Non è
pedanteria — è ciò che rende la cosa vettorizzabile in torch e riproducibile a parità di
seed. Uno `scatter` non deterministico su GPU basta a rendere impossibile il debug di
una regressione.

```
per ogni step:
    5.1  percezione        (legge pheromone, nutrient — mai error)
    5.2  rotazione
    5.2b propriocezione    (run-and-tumble sul proprio gain — non sul campo)
    5.3  avanzamento
    5.4  deposito          -> canvas, pheromone      [modifica il quadro]
    5.5  consumo           -> energy                 [l'unico punto col target]
    5.6  costi             -> energy
    5.7  morte
    5.8  riproduzione
    5.9  invecchiamento del pigmento
    5.10 respiro dei campi
```

---

### 5.1 Percezione

Tre sensori davanti all'agente, a distanza `sensor_distance` e agli angoli
`−sensor_angle`, `0`, `+sensor_angle` rispetto a `dir`. Ciascuno campiona un **campo
di attrattività**:

```
A = pheromone + w_nutrient · nutrient
S_L, S_C, S_R = campiona(A, punti dei tre sensori)
```

Il campionamento è bilineare (`grid_sample`): con quello nearest gli agenti si
incastrano sulla griglia dei pixel e le tracce diventano scalinate.

| Alternativa | Effetto | Costo |
| --- | --- | --- |
| ★ 3 sensori, bilineare | classico Physarum, tracce filamentose e coerenti | 3 letture/agente |
| 5 sensori | virate più fluide, meno oscillazione | +66% letture |
| Anello continuo (8-16 direzioni, softmax) | moto quasi olonomo, perde il carattere "traccia" | caro |
| `A` con termine di affollamento `− w_crowd · density` | gli agenti si evitano, copertura più uniforme, meno grumi | serve un campo `density` in più (uno scatter) |

Il termine di affollamento è l'unica alternativa che vale la pena tenere dietro flag fin
da subito: il difetto tipico di Physarum + selezione è che tutti si ammassano dove il
cibo è stato appena trovato, e chi arriva secondo mangia zero. Un `w_crowd` piccolo
distribuisce la fame.

**Nota di principio:** qui è dove va messo `guidance` se un giorno lo si vuole
assaggiare — `A += guidance · error`. A `guidance = 0` (default) gli agenti sono ciechi
al proprio errore. Ogni valore > 0 è un passo verso il barare, ed è esposto solo per
poter *misurare* quanto costa l'onestà. È **lettura spaziale** dell'errore, quindi cade
dal lato sbagliato della linea di §2.2 — a differenza della propriocezione di §5.2b, che
cade da quello giusto.

---

### 5.2 Rotazione

| Alternativa | Regola | Carattere visivo |
| --- | --- | --- |
| Physarum classico | se `S_C` è il massimo resta dritto; se è il minimo gira a caso di `±rotation_angle`; altrimenti gira verso il maggiore fra L e R, sempre di `rotation_angle` | nervoso, a scatti, molto "muffa" |
| ★ Proporzionale | `Δθ = rotation_angle · (S_R − S_L) / (S_L + S_C + S_R + ε)` | virate morbide, curve continue — **legge come pennellata** |
| Softmax stocastica | scegli fra i tre angoli con `p ∝ exp(S/T)` | esplorazione migliore, traccia più sporca |

A qualunque delle tre si somma sempre un rumore angolare `θ += 𝒩(0, angle_noise)`.
Senza, gli agenti in campo piatto vanno dritti all'infinito e la copertura è a strisce.

La scelta proporzionale non è cosmetica: lo stadio 3 deve produrre **pennellate**, e la
rotazione a scatti del Physarum canonico produce filamenti spezzati. Il flag per il
classico resta perché su `fitness_scale` grossolani il look nervoso potrebbe essere
esattamente quello che serve.

---

### 5.2b Propriocezione — *run and tumble*

Il pezzo che rende efficiente il **singolo agente** invece della sola popolazione. Senza
di questo, nessun individuo cerca: si limita a stare dove è nato (§2.1).

L'agente non vede l'errore, ma ricorda quanto ha mangiato allo step precedente. È
sufficiente per la strategia di ricerca dei batteri — la chemiotassi di *E. coli*:

> **Se sto mangiando vado dritto; se non sto mangiando giro a caso.**

```
# gain viene calcolato in 5.5, quindi qui si guardano i due step già conclusi
Δ = gain_i[t−1] − gain_i[t−2]                  # sto andando sempre meglio?
se Δ < tumble_threshold:
    θ += 𝒰(−tumble_angle, +tumble_angle)       # tumble: virata casuale ampia
altrimenti:
    (rotazione normale di 5.2)                 # run: prosegui
```

Il ritardo di uno step è ininfluente — è lo stesso che ha il batterio, che confronta la
concentrazione di adesso con quella di un attimo fa.

Due righe. L'agente non sa dove sia il cibo: allunga le corse nelle direzioni che pagano
e le accorcia in quelle che non pagano. È una stima del gradiente per differenze finite
lungo la propria traiettoria, **senza mai calcolare un gradiente e senza mai vedere il
campo**. Non serve memoria: basta conservare `gain` dello step precedente, un tensore
`(N,)` in più.

Perché è legittimo: usa solo `gain_i`, cioè quanto *l'agente stesso* ha guadagnato. È
lettura temporale della propria riuscita, non lettura spaziale della soluzione — il lato
buono della linea di §2.2. È lo stesso canale di veto della selezione, spostato dentro la
vita del singolo.

| Alternativa | Effetto | Verdetto |
| --- | --- | --- |
| ★ Run-and-tumble su `Δgain` | ricerca individuale vera, gratis, non barante | attivare fin dal lab, dietro `tumble_enabled` |
| Su `gain` assoluto invece che su `Δ` | più semplice, ma un agente in zona buona ma satura continua dritto all'infinito | no |
| Media mobile di `gain` su k step | meno rumoroso, reagisce più tardi | flag, se il segnale a un solo step risulta troppo sporco |
| Modulazione della `speed` invece della virata | l'agente rallenta dove mangia invece di girare: si sofferma, deposita di più. Effetto visivo diverso e forse migliore | vale un confronto in `compare` |

**Il punto per cui è decisivo:** finché i suoi parametri (`tumble_threshold`,
`tumble_angle`) sono costanti di configurazione, questa resta una strategia di specie.
Quando passeranno nel genoma comportamentale (§14.1) diventeranno ereditabili — **e solo
allora i figli saranno letteralmente più bravi a cercare del genitore.** È il pezzo che
chiude il cerchio fra i due livelli di §2.1.

---

### 5.3 Avanzamento

```
pos_prev = pos
pos += speed · (cos θ, sin θ)
```

**Bordi del canvas:**

| Alternativa | Effetto |
| --- | --- |
| Wrap (toro) | classico Physarum, ma qui teletrasporta pigmento da un lato all'altro del soggetto: **da escludere** |
| ★ Riflessione | `θ` specchiato sulla normale del bordo; l'agente rimbalza e resta vivo |
| Clamp + virata casuale | funziona, ma gli agenti strisciano lungo il bordo formando una cornice |
| Morte | pulito ma spreca genoma proprio dove il tessuto finisce |

**Bordi del tessuto** (`nutrient` sotto soglia) — è la domanda più interessante delle
due, perché decide se lo sciame può uscire dalla creatura:

| Alternativa | Effetto |
| --- | --- |
| ★ Costo extra fuori dal tessuto (`starvation_cost`) | barriera morbida: si può sconfinare, ma si paga. Le escursioni brevi restano possibili e producono le sbavature che rendono viva la silhouette |
| Muro duro (riflessione su `nutrient`) | silhouette nettissima, sciame ingabbiato, bordo che sembra ritagliato |
| Nessun vincolo | lo sciame colonizza lo sfondo, dove il target è uniforme e l'errore è quasi zero: muoiono da soli, ma nel frattempo sporcano |

La barriera morbida è coerente col principio: non si vieta, si rende costoso. È
selezione, non un `if`.

---

### 5.4 Deposito — la pennellata

**È il passo che modifica il quadro.** Tutto il resto serve a decidere chi lo esegue.

La traccia è il segmento `pos_prev → pos`. Si campiona in `K = ceil(‖Δpos‖ / 0.5)` punti
e attorno a ogni punto si stampa un pennello di raggio `brush_radius`.

| Alternativa di pennello | Resa |
| --- | --- |
| Punto singolo (nearest) | puntinismo, aliasing, gratis |
| Disco pieno | tratto uniforme a bordi duri, come l'attuale `_splat` ([refiner.py:340](particles/refiner.py:340)) |
| ★ Kernel gaussiano `σ = brush_radius/2` | bordi morbidi che si sovrappongono in strati: acquerello. Il costo è `K · (2r+1)²` scritture, dominabile tenendo `brush_radius ≤ 2` alla risoluzione di lavoro |

Il blending. Ogni campione `i` porta un peso `a_i(p) = deposit_alpha · kernel(p)` sul
pixel `p`. Il modo ingenuo — scrivere in sequenza `canvas = (1−a)·canvas + a·gene` — non
è order-independent: su GPU due agenti sullo stesso pixel danno risultati diversi a ogni
run. La forma corretta accumula e poi risolve una volta sola:

```
W(p) = Σ_i a_i(p)                      # scatter_add, peso totale
C(p) = Σ_i a_i(p) · gene_i             # scatter_add, colore pesato
α(p) = min(W(p), 1)
canvas(p) = (1 − α)·canvas(p) + α · C(p)/max(W(p), ε)
```

Due agenti con geni diversi sullo stesso pixel depositano la **media** dei loro colori,
pesata per quanto ci hanno messo. È deterministico, è una sola passata, ed è anche più
giusto: nessuno dei due "vince" per caso di scheduling.

Contemporaneamente `pheromone(p) += Σ_i deposit_pheromone · kernel(p)`.

**Larghezza del tratto:** costante nella versione base. Diventa un gene
nell'estensione comportamentale (§14) — ed è lì che nasce la specializzazione della
pennellata.

---

### 5.5 Il consumo — dall'errore all'energia

Il punto che meritava chiarimento. La parola "consumo" è letterale, non metaforica:

> **Il campo `error` è lo stock di cibo. Dipingere lo converte in energia, e nel farlo
> lo esaurisce.** L'errore mangiato non c'è più: il pixel corretto non nutrirà nessun
> altro. Non è una metafora appiccicata sopra a un'ottimizzazione — è la stessa
> quantità, letta due volte.

Meccanica in tre tempi.

**1. Errore prima e dopo.**

```
err_before = ‖canvas_prima − target‖   (per pixel, OKLab)
... deposito (5.4) ...
err_after  = ‖canvas_dopo  − target‖
gain(p) = err_before(p) − err_after(p)          # > 0 = miglioramento
```

Si ricalcola il campo intero, non solo i pixel toccati: a 256²-512² è una sottrazione
fusa che costa meno del masking necessario a evitarla.

**2. Attribuzione.** Il problema serio: su un pixel toccato da più agenti, di chi è il
merito? Lo stesso peso `a_i(p)` che ha deciso il colore decide anche la spartizione:

```
gain_i = Σ_p [ a_i(p) / W(p) ] · gain(p)
```

Proprietà che conta: `Σ_i gain_i = Σ_p gain(p)` — **conservazione esatta**. L'errore
tolto viene pagato una volta sola e diviso fra chi l'ha tolto. Da qui esce la
competizione: due agenti sullo stesso pixel dimezzano la razione, e sopravvive chi
trova pixel che nessun altro sta già sistemando. La spinta a disperdersi non è
programmata, viene dalla contabilità.

| Alternativa di attribuzione | Effetto | Verdetto |
| --- | --- | --- |
| ★ Ripartizione proporzionale ad `a_i` | conservativa, genera competizione locale | default |
| Winner-takes-all (agente con `a` massimo) | differenziazione più rapida e brutale, energia molto rumorosa | flag |
| Controfattuale (per ogni agente, il guadagno che *avrebbe* da solo) | niente conflitti, ma più agenti "mangiano" lo stesso errore → sovralimentazione e popolazione che esplode | solo come baseline di debug |
| Shapley | corretto, combinatorio | fuori discussione |

**3. Metrica dell'errore.**

```
e(p) = ‖canvas(p) − target(p)‖₂       in OKLab
e_eff(p) = max(0, e(p) − tolerance)   # sotto tolleranza non nutre più
```

| Alternativa | Effetto |
| --- | --- |
| ★ Norma L2 in OKLab | distanza percettiva, comportamento uniforme su tutto lo spazio colore |
| L2 al quadrato | coda pesante: premia molto chi corregge errori grossi, ignora i piccoli → resa più contrastata, meno rifinita |
| Pesi diversi su L e (a,b) | separa "sbagliare la luminanza" da "sbagliare la tinta". Pesare di più la luminanza dà un quadro che legge bene anche in bianco e nero |

`fitness_scale` agisce qui: a `1` (deciso, §16) l'errore si misura a piena risoluzione di
lavoro. A valori > 1 si confrontano canvas e target *downsamplati*, e allora
l'attribuzione del passo 2 accumula i pesi `a_i` nelle celle grosse invece che nei pixel.
Meccanica identica, solo su una griglia più rada — ma il codice deve prevederlo dal
principio, perché è la manopola che porta la resa da fedele a impressionista.

---

### 5.6 I costi

```
energy += gain_scale · gain_i
        − cost_deposit · Σ_p a_i(p)          # proporzionale al pigmento steso
        − cost_life                          # fisso per step
        − starvation_cost · [fuori dal tessuto]
```

Il costo del deposito è **proporzionale alla massa depositata**, non fisso per step. È
il dettaglio che chiude la degenerazione golosa: ridipingere un pixel già corretto rende
zero e costa pieno, quindi è a somma negativa. Con un costo fisso, un pennello grande
sarebbe energia gratis e tutti diventerebbero rulli.

| Aggiunta | Perché | Verdetto |
| --- | --- | --- |
| ★ Tetto all'energia (`energy_cap`) | senza, un agente fortunato accumula una riserva enorme e diventa immortale in una zona ormai sterile | tenere |
| ★ Senescenza (`cost_life` cresce con `age`, o `max_age` netto) | garantisce ricambio generazionale anche a regime | tenere, è ciò che mantiene vivo il processo infinito (§7) |
| Costo proporzionale a `speed` | i veloci pagano di più; ha senso solo col genoma comportamentale, dove `speed` è ereditabile | rimandare a §14 |

---

### 5.7 Morte

`energy < 0` → `alive = False`, lo slot si libera.

Soglia netta e non probabilistica: è più leggibile a schermo e più facile da tarare.

**Il pavimento di popolazione.** Se i vivi scendono sotto `min_population`, gli slot
vuoti vengono ripopolati. Qui c'è una trappola di principio da evitare esplicitamente:

> **Il ripopolamento non può guardare `error`.** Far nascere gli agenti dove l'errore è
> alto sarebbe esattamente il barare che tutto il documento evita, e per giunta mascherato
> da dettaglio implementativo.

| Alternativa di ripopolamento | Legittimità |
| --- | --- |
| ★ Posizione campionata da `nutrient`, gene = canvas locale + mutazione | legittima: `nutrient` e `canvas` sono prodotti nostri, non ground truth |
| Posizione e gene da un sopravvissuto scelto a caso | legittima, conserva il genoma; tende a impoverire la diversità |
| Posizione dove `error` è alto | **vietata** |

---

### 5.8 Riproduzione

Candidato: `alive & energy > reproduction_threshold`. Se ci sono slot liberi:

```
energy_genitore /= 2
figlio:  pos    = pos_genitore + jitter(birth_jitter)
         dir    = dir_genitore + 𝒩(0, birth_dir_sigma)
         gene   = clamp_gamut( gene_genitore + 𝒩(0, mutation_sigma) )   # in OKLab
         energy = energy_genitore                                       # la metà
         age    = 0
```

La fissione per dimezzamento è la scelta giusta perché rende la riproduzione **a somma
zero sull'energia**: nessuna materia dal nulla, e un genitore che si riproduce troppo si
indebolisce da solo. È un freno automatico.

**Assegnazione degli slot** — servono meno posti liberi che candidati, quasi sempre:

| Alternativa | Effetto |
| --- | --- |
| Elitaria (i candidati con più energia) | convergenza rapida, diversità che collassa presto, territori netti |
| ★ Casuale fra gli eleggibili | la soglia fa già da filtro; la casualità sull'ordine tiene aperta l'esplorazione |
| Torneo a k | via di mezzo, un parametro in più da tarare |

**Mutazione** — è il generatore di tutte le proposte del sistema, quindi la sua forma
decide che quadro esce:

| Alternativa | Resa |
| --- | --- |
| ★ Gaussiana isotropa in OKLab | uniforme e prevedibile, il default sensato |
| Anisotropa (σ piccolo su L, grande su a,b) | i valori restano giusti e varia la tinta: **il look "dipinto"** — un quadro tonalmente corretto e cromaticamente libero |
| Coda pesante (Cauchy) | rari salti grossi: esce dai minimi locali, produce lampi di colore sbagliato che poi muoiono. Visivamente vivace |
| Snap alla palette (§e di PLAN) | il gene si aggancia alla voce di palette più vicina: resa serigrafica, spazio di ricerca discreto e piccolissimo |

L'anisotropa e lo snap-a-palette sono i due esperimenti che mi aspetto diano più
carattere; la gaussiana isotropa resta il riferimento contro cui misurarli.

**Crossover** (due genitori, gene = media pesata) — su un vettore di 3 numeri aggiunge
pochissimo, e la media di due colori vicini è un colore vicino. Registrato e rimandato.

---

### 5.9 L'invecchiamento del pigmento — `DECISO: mai cristallizzazione`

Secondo livello di selezione: non solo chi dipinge, ma anche **cosa resta dipinto**.

Deciso in §16: il processo è idealmente infinito, quindi la vita del colore è
inversamente proporzionale all'errore ma **non arriva mai a zero decadimento**.

```
λ(p) = λ_min + (λ_max − λ_min) · saturate( e(p) / e_ref )
canvas(p) ← canvas(p) + λ(p) · ( base(p) − canvas(p) )
```

- errore alto → `λ ≈ λ_max` → il pigmento sbagliato svanisce in fretta verso il tessuto NCA;
- errore basso → `λ ≈ λ_min` → il pigmento giusto dura moltissimo, ma **non per sempre**.

`λ_min` è il parametro che rende il processo infinito, e vale la pena dire esattamente
cosa fa: è il **ricambio metabolico del quadro**. A `λ_min = 0` si ricade nella
cristallizzazione (il quadro si fissa, lo sciame muore di fame, fine); a `λ_min > 0` il
quadro perde lentamente definizione ovunque, il che **rigenera cibo di continuo** e
tiene viva la popolazione.

Va detto chiaro il prezzo, perché è reale: **l'errore non andrà mai a zero.** Si assesta
a un equilibrio dove il tasso di riparazione dello sciame eguaglia il tasso di
degradazione `λ_min`. Se `λ_min` è troppo alto il quadro non arriva mai a definirsi e
sfarfalla; troppo basso e la popolazione si riduce a pochi custodi. È la seconda
manopola da tarare dopo `tolerance`, e la diagnostica di §12 serve proprio a vederla.

Contropartita, che è poi il motivo per cui l'hai chiesto: **il quadro finito non è
immobile.** Ha un brulichio residuo, una squadra di manutenzione perpetua, e non c'è
nessun istante in cui "è finito". Il video può durare quanto si vuole.

| Alternativa di decadimento | Resa |
| --- | --- |
| ★ Rilassamento esponenziale verso `base` | il pigmento sbagliato "riaffonda" nel tessuto NCA; sempre leggibile |
| Verso lo sfondo | il quadro si buca visibilmente dove è sbagliato: drammatico, rischioso |
| Sfocatura locale (`λ` come forza di un blur) | il colore non svanisce, **si scioglie**. Molto pittorico, e riapre errore in modo più morbido |
| Contatore di vita per pixel con scadenza netta | è la lettura letterale di "span vitale", ma produce popping visibile e costa un campo in più |

Il blur locale è l'alternativa che vale la pena implementare per seconda: è la stessa
riga di codice con un operatore diverso, e la resa "a olio ancora fresco" potrebbe
battere il rilassamento.

---

### 5.10 Il respiro dei campi

```
pheromone = blur3x3(pheromone) · evaporation
nutrient  = alpha dell'NCA al frame corrente        # nel lab: statico
error     = ricalcolato per lo step successivo
```

L'ordine conta: la diffusione va *dopo* i depositi dello step, così il feromone che un
agente lascia è già spalmato quando qualcun altro lo annusa al passo successivo.
`evaporation` è la memoria del sistema — è il parametro che decide se lo sciame disegna
tracce persistenti o si comporta da nuvola nervosa.

---

## 6. I due livelli di selezione

Selezione **sugli agenti** (chi vive, §5.7-5.8) e selezione **sui depositi** (cosa resta,
§5.9). Il secondo livello è la lettura letterale dello "span vitale per pixel", e serve a
tre cose:

1. **il canvas si autopulisce** — aloni, tratti fuori posto e sporco svaniscono da soli:
   il difetto visivo numero uno dello stadio attuale sparisce per costruzione, senza
   doverlo inseguire;
2. **la pittura sbagliata libera cibo** — un tratto che decade riapre errore in quella
   zona, quindi la richiama lo sciame: il sistema si ripara;
3. **il quadro non si ferma mai** — con `λ_min > 0` anche il pigmento giusto si consuma
   lentissimamente, e questo è ciò che rende il processo perpetuo invece che terminale.

Anche qui la pressione evolutiva non dipinge: il decadimento **toglie** informazione
sbagliata, non ne mette di giusta.

---

## 7. L'arco narrativo, che non va programmato

Emerge dalle regole, e questo è il motivo per cui lo stadio 3 diventa guardabile:

1. **Colonizzazione.** Canvas povero di alta frequenza → quasi ogni deposito migliora
   qualcosa → tutti mangiano, la popolazione esplode.
2. **Territori.** Le stirpi si radicano dove il loro colore paga; i confini fra
   territori si assestano lungo i bordi cromatici del soggetto. Si vedono le frontiere
   contendersi i pixel.
3. **Carestia.** Le zone finite non rendono più: chi ci resta muore di fame. Lo sciame
   **si dirada dove l'immagine è pronta e si addensa dove manca**. Nessuno lo programma:
   è la fitness a miglioramento marginale che lo produce.
4. **Regime.** Qui l'arco cambia rispetto alla prima stesura. Non c'è estinzione: la
   popolazione si assesta su un numero basso e stabile, quel tanto che basta a
   compensare `λ_min`. Il quadro è leggibile e **continua a respirare**: qualche stirpe
   superstite ripassa in eterno i punti che si consumano.

L'NCA continua a girare sotto, quindi il tessuto si muove e sposta di continuo il
bersaglio: un altro motivo per cui il lavoro non finisce mai.

**Conseguenza sul montaggio, da mettere a verbale:** senza estinzione, il video non ha
più un finale *automatico*. Il finale torna a essere una decisione di regia — un hold, un
fermo immagine, una chiusura di camera. In cambio si guadagna qualcosa che prima non
c'era: la sequenza è **potenzialmente loopabile**, perché lo stato a regime è
statisticamente stazionario.

---

## 8. Il clima — la regia dell'evoluzione

Problema pratico: l'arco deve stare dentro i secondi assegnati. Sperare che si assesti da
sé nel tempo giusto è ingenuo.

Non si forza il risultato, si **modula l'ambiente** nel tempo — cibo abbondante e costo
di vita basso all'inizio (esplosione), costo crescente verso la fine (carestia
anticipata, arco compresso). È l'unica leva di regia che non sporca il principio: si
cambia il clima, non si suggerisce la risposta.

Parametri `climate.*`: curve nel tempo su `cost_life`, `gain_scale` e `mutation_sigma`.
Mutazione alta all'inizio (esplorazione) e bassa alla fine (rifinitura) è il classico
*annealing*, e qui ha anche una lettura visiva: si passa da pennellate audaci a ritocchi.

**Deciso** (§16): ogni voce di `climate` accetta sia una curva sia una costante. La
forma è la stessa — `{start, end, easing}` con `start == end` che significa "costante" —
così si passa da clima a nessun clima cambiando un numero, e il confronto A/B è
immediato. Le curve riusano [easing.py](rendering/easing.py), che esiste già.

**Sul tempo di simulazione** — nota operativa concordata: se l'evoluzione ha bisogno di
più step di quanti ne entrino nei secondi previsti, **si simula più a lungo e si accelera
in riproduzione**. `sim_steps` e `fps` sono parametri indipendenti, e un
`step_stride` (scrivi un frame ogni k step) permette di rendere 3000 step di simulazione
in 300 frame di video. Costa solo tempo di calcolo, e non tocca nessuna regola.

---

## 9. Il genoma — `DECISO: prima solo colore`

**Versione base — solo colore.** Il gene è un punto in OKLab, tre numeri. Spazio di
ricerca piccolo, converge in fretta, effetto visivo già completo. È da qui che si parte.

L'estensione al comportamento è rimandata a §14 come miglioria futura, da attivare
quando la base è calibrata.

---

## 10. Spazio colore — `DECISO: OKLab`

Geni, distanze, blending e decadimento in **OKLab**, non in RGB. Due motivi concreti:

- in RGB una mutazione di ampiezza fissa produce salti percettivi molto diversi a seconda
  della zona di colore; in OKLab `mutation_sigma` significa la stessa cosa ovunque, e la
  ricerca è uniforme;
- `tolerance` diventa una soglia **percettiva**: "abbastanza simile per l'occhio", che è
  ciò che vogliamo davvero, invece di "abbastanza simile per la somma dei quadrati".

Il target si converte una volta all'avvio; il canvas *vive* in OKLab e si converte in
sRGB solo alla scrittura del frame. Serve un `clamp_gamut` sui geni mutati, perché
OKLab contiene coordinate che non hanno un sRGB corrispondente: senza, l'evoluzione
scopre in fretta i colori impossibili e li usa per barare sull'errore.

---

## 11. Il laboratorio — `DECISO: si parte da qui`

Lo stadio 3 **non** si integra subito con SCA e NCA. Ci sono troppi parametri accoppiati
e troppe varianti da confrontare: farlo dentro la pipeline completa significa aspettare
un render intero per giudicare una manopola. Si isola, si tara, e solo dopo si collega.

### 11.1 Il surrogato dell'NCA

Il lab non ha bisogno dell'NCA per essere fedele, e questo è il punto che lo rende
possibile. Lo stadio 3 riceve dallo stadio 2 un'immagine **cui manca l'alta frequenza**:
la si fabbrica in una riga.

```
base = upsample( downsample(target, 128), work_size )      # esattamente il difetto dell'NCA
nutrient = alpha(base) smussato
```

Il canvas parte da `base`, come in produzione. Le condizioni di lavoro dello sciame — che
tipo di errore ha davanti, quanto ne ha — sono quelle vere.

Con un'onestà da tenere presente: l'output NCA vero non è solo sfocato, ha anche errori
di colore e di forma suoi. Quindi il surrogato prevede due sporcature opzionali,
`surrogate_color_noise` e `surrogate_posterize`, per verificare che la taratura regga
anche quando il punto di partenza è *sbagliato* e non solo *molle*.

### 11.2 Modalità di lavoro

Tre, sullo stesso motore:

**`live`** — finestra OpenCV, simulazione in corso, trackbar sui parametri caldi
(`deposit_alpha`, `mutation_sigma`, `cost_life`, `evaporation`, `tolerance`,
`brush_radius`). Tasti: pausa, avanzamento a singolo step, reset a parità di seed, dump
dello stato corrente su disco. Cambiare un parametro **non** riavvia: si vede lo sciame
reagire, che è il modo più veloce per capire cosa fa una manopola.

Vista commutabile con i tasti numerici — sono cinque letture dello stesso istante:
`canvas` · `error` (falsi colori) · `pheromone` · densità di popolazione · mappa dei geni.

**`sweep`** — headless. Griglia su 1-2 parametri, N run a seed fissato, un **contact
sheet** PNG con le miniature finali etichettate più un CSV con le metriche di §12. È il
modo per scegliere fra le alternative di §5 senza affidarsi alla memoria di com'era il
run precedente.

**`compare`** — due configurazioni affiancate nella stessa finestra, stesso seed, stesso
target. Serve esattamente ai confronti binari che il documento continua a rimandare:
rotazione classica contro proporzionale, mutazione isotropa contro anisotropa,
decadimento verso `base` contro blur.

### 11.3 Dimensionamento

Deciso di partire piccolo, come chiesto:

| | Lab (iterazione) | Produzione |
| --- | --- | --- |
| Risoluzione di lavoro | 256² | 512² |
| Agenti | 4 096 | 20-50 000 |
| Step | 600-1 000 | 3 000+ |
| Obiettivo | ≥ 30 step/s interattivi | offline |

**GPU.** Tutto il ciclo è vettorizzato in torch e gira indifferentemente su CPU o CUDA
(`device` in config, con il fallback che c'è già in
[refiner.py:16](particles/refiner.py:16)). Un'avvertenza contro-intuitiva ma concreta: a
4 096 agenti su 256² il lavoro per step è piccolo e il tempo è dominato dall'overhead di
lancio dei kernel — la CPU può risultare **più veloce** della GPU in modalità `live`. Il
lab misura e stampa gli step/s, e si sceglie il device sulla misura, non sull'aspettativa.
La GPU serve senz'altro in `sweep` (run in parallelo sul batch) e in produzione.

Il resto è igiene: seed esplicito in config, ogni run scrive accanto all'output il dump
completo della configurazione, e la simulazione è riproducibile bit a bit a parità di
seed e device — che è tutto il valore che ha una suite di controllo.

### 11.4 Dove sta il codice

```
swarm/
  agents.py       # stato, percezione (5.1), movimento (5.2-5.3), deposito (5.4)
  selection.py    # consumo (5.5), costi (5.6), morte (5.7), riproduzione (5.8)
  fields.py       # feromone, nutriente, errore, invecchiamento del pigmento (5.9-5.10)
  simulation.py   # il ciclo, il clima, le metriche
  lab.py          # live / sweep / compare
config/
  swarm_config.py # tutti i parametri, con i default di §15
```

L'integrazione con la timeline (PLAN §3.3) tocca solo `simulation.py`, che al posto di
un `nutrient` statico riceve l'alpha dell'NCA al frame corrente. È deliberatamente
l'unica cucitura.

---

## 12. Diagnostica — lo sciame ha una sua curva di apprendimento

Poiché è un'ottimizzazione vera, produce metriche vere. Da salvare in
`outputs/swarm/<nome>_evolution.png`:

- errore medio nel tempo → **una loss curve**, esattamente come
  [nca_loss.png](outputs/nca/jellyfish_loss.png);
- popolazione viva nel tempo → boom, carestia, assestamento;
- diversità genetica (varianza dei geni) nel tempo → esplorazione che si chiude in
  convergenza;
- **energia totale e guadagno per step** → è la lettura diretta di quanto cibo resta;
  a regime deve oscillare attorno a una costante, e se scende a zero `λ_min` è troppo
  basso.

L'ultima metrica sostituisce la "frazione di pixel cristallizzati" della prima stesura,
che senza cristallizzazione non vuol più dire niente.

Vale anche come argomento: **lo stadio 3 avrà una curva di apprendimento come ce l'ha
l'NCA.** È la prova che non è un effetto grafico.

**Aggiunta dopo l'implementazione.** Queste metriche guardano la *popolazione* e non
bastano a giudicare il *quadro*: non distinguono un run che ha ridipinto il soggetto da
uno che l'ha spalmato, né una pennellata dal rumore. Le metriche di immagine
(fedeltà mascherata al tessuto, dettaglio, coerenza del tratto, croma, sfarfallio), i
criteri di successo che ne derivano e la suite che tara i parametri da sola stanno in
[Swarm_Tuning.md](docs/Swarm_Tuning.md).

---

## 13. Come può fallire

| Rischio | Perché | Contromisura |
| --- | --- | --- |
| **Fotocopiatrice** | converge a errore zero: un modo costoso di copiare il target | `tolerance` e `fitness_scale`: il risultato interessante è *come* approssima, non *quanto*. **Primo parametro da tarare.** |
| **Degenerazione golosa** | agenti fermi che ridipingono lo stesso pixel giusto | fitness a miglioramento *marginale* + costo di deposito proporzionale alla massa (§5.6) |
| **Estinzione precoce** | costo troppo alto: muoiono tutti prima di combinare qualcosa | riserva di energia alla nascita, `climate` generoso all'inizio, `min_population` (§5.7) |
| **Equilibrio rumoroso** | `λ_min` troppo alto: il quadro brulica e non si definisce mai | è il prezzo del processo infinito; si vede subito nella curva di errore (§12) e si tara `λ_min` contro `tolerance` |
| **Ammassamento** | tutti sul feromone più forte, e chi arriva secondo mangia zero | termine di affollamento nei sensori (§5.1) |
| **Non determinismo** | scatter concorrenti su GPU → run irriproducibili, debug impossibile | accumulazione order-independent in §5.4-5.5, imposta fin dal primo commit |
| **Tuning permaloso** | i parametri Physarum sono notoriamente sensibili | è la ragione d'essere del laboratorio (§11) |
| **Deriva cromatica** | mutazione senza pressione sufficiente → colori slavati | mutazione in OKLab + annealing di `mutation_sigma` + `clamp_gamut` |

Costo computazionale: il campo errore si ricalcola ad ogni step — una sottrazione a piena
risoluzione, banale su GPU. Popolazione 10-50k agenti interamente vettorizzata in torch,
lo stesso ordine di grandezza dell'attuale [refiner.py](particles/refiner.py).

---

## 14. Migliorie future

Registrate qui perché sono buone idee fuori dallo scopo della prima implementazione.
Nessuna di queste va scritta finché §5 non funziona e non è tarata.

### 14.1 Genoma comportamentale — la pennellata che si specializza

Il gene porta anche `speed`, `sensor_angle`, `sensor_distance`, `deposit_alpha`,
`brush_radius` e **i due parametri di run-and-tumble** (§5.2b). Questi ultimi sono i più
interessanti dei sette, perché sono gli unici che riguardano la *strategia di ricerca*:
finché sono costanti di config, cercare bene è una dote di specie; da ereditabili in poi,
**i figli diventano davvero più bravi a trovare cibo del genitore** — il che è l'unico
modo di chiudere il divario descritto in §2.1 fra evoluzione del colore (vera) ed
evoluzione della posizione (solo demografia).

Conseguenza sull'aspetto: **regioni diverse evolvono pennellate diverse**. Le punte
dei tentacoli premiano agenti sottili e veloci, la campana agenti larghi e lenti.
Specializzazione emergente della pennellata, senza averla scritta da nessuna parte — ed è
la sola strada per cui il quadro finale ha una *mano* diversa in punti diversi.

Perché è rimandata e non scartata: lo spazio di ricerca passa da 3 a 8 dimensioni, la
convergenza rallenta, e soprattutto **non si distingue più** se un risultato brutto viene
dai parametri sbagliati o dall'evoluzione che non ha avuto tempo. Serve una base tarata
come termine di paragone.

Note per quando si farà: ogni gene comportamentale ha bisogno del proprio `mutation_sigma`
(le scale sono incomparabili) e di un range di clamp; e `cost_life` dovrà dipendere da
`speed` e `brush_radius`, altrimenti l'evoluzione scopre subito che "grande e veloce" è
gratis e collassa tutta la popolazione su quel fenotipo.

### 14.2 Altre, in ordine di interesse

- **Decadimento come sfocatura** (§5.9) — il pigmento sbagliato si scioglie invece di
  svanire. Una riga diversa, resa potenzialmente molto migliore.
- **Mutazione anisotropa** (§5.8) — luminanza conservativa, cromia libera. Il candidato
  più promettente per un look pittorico.
- **Snap alla palette** (§5.8) — gene agganciato alla palette estratta (PLAN §e): resa
  serigrafica e spazio di ricerca minuscolo.
- **Loop perfetto** — a regime lo stato è stazionario (§7): con un po' di lavoro sul
  matching dei frame la sequenza finale può chiudersi su sé stessa.
- **Fitness sull'output NCA invece che sul target** — la variante severa già registrata
  in [PLAN.md](docs/PLAN.md): lo stadio 3 non tocca mai il ground truth. Tetto di qualità
  più basso, purezza concettuale massima. Costa un flag.
- **Crossover** (§5.8) — poco promettente su un gene di 3 numeri, diventa sensato solo
  col genoma comportamentale.

### 14.3 Neuroevoluzione — valutata e scartata

L'idea naturale una volta capito §2.1: dare a ogni agente una micro-rete (3 input → 4
nascosti → 2 output), i pesi come genoma, ~30 numeri da evolvere. Il cervello vero.

**Scartata, e per un motivo strutturale, non per prudenza.** Il collo di bottiglia non è
la rete, sono gli input: un agente riceve solo feromone, nutriente e il proprio `gain`.
Con tre canali non c'è quasi niente da imparare che il run-and-tumble di §5.2b non faccia
già in due righe — e le poche politiche in più che una rete potrebbe esprimere ("evita la
folla", "rallenta sul tessuto denso") sono più economiche come termini espliciti nei
sensori. In cambio lo spazio di ricerca passa da 5 a ~35 dimensioni e chiede migliaia di
generazioni che non abbiamo.

Registrata qui perché la conclusione è utile anche in negativo: **in questa architettura
il limite non è la capacità di calcolo degli agenti, è quanto poco possono percepire.** E
percepire di più significa avvicinarsi a `guidance`, cioè al barare.

---

## 15. Parametri di regia

Ordinati per impatto visivo, non per posizione nel codice. La colonna "lab" è il punto di
partenza proposto per §11, non un valore tarato.

| Parametro | Cosa governa | Lab | Effetto se lo alzi |
| --- | --- | --- | --- |
| `fitness_scale` | scala a cui si misura l'errore | `1` | **da fedele a impressionista** — errore su immagine ridotta ⇒ pennellate larghe e gestuali |
| `tolerance` | quando smettere di premiare | ~`0.02` | ricostruzione più stilizzata, si ferma prima |
| `decay_min` (`λ_min`) | ricambio metabolico del quadro | piccolo, > 0 | quadro più vivo e più instabile; a 0 si cristallizza |
| `decay_max` (`λ_max`) | quanto in fretta sparisce il pigmento sbagliato | | autopulizia più aggressiva |
| `deposit_alpha` | opacità del singolo tratto | ~`0.15` | da acquerello stratificato a tempera piena |
| `brush_radius` | larghezza della pennellata | `1-2` | tratto grasso, meno dettaglio, più costo |
| `mutation_sigma` | ampiezza dell'esplorazione | | più varietà cromatica, convergenza più lenta |
| `sensor_angle` / `sensor_distance` | forma delle tracce | | reti larghe e ramificate vs filamenti stretti |
| `rotation_angle` | reattività della virata | | tracce nervose vs curve ampie |
| `tumble_enabled` / `tumble_threshold` / `tumble_angle` | ricerca individuale (§5.2b) | on | l'agente si intestardisce meno e cambia zona prima |
| `birth_jitter` | **tasso di mutazione della posizione** (§2.1) | piccolo | stirpi che esplorano invece di radicarsi; a 0 i territori si fossilizzano |
| `evaporation` | memoria del feromone | | tracce persistenti e ordinate vs sciame nervoso |
| `population_cap` / `min_population` | densità e pavimento | `4096` / ~5% | copertura veloce vs rada e "disegnata" |
| `gain_scale`, `cost_life`, `cost_deposit` | il metabolismo | | vedi §5.6 |
| `reproduction_threshold` | quanto si deve mangiare per figliare | | popolazione più selettiva, ricambio più lento |
| `w_nutrient`, `w_crowd` | attrattività del tessuto, repulsione | | aderenza alla creatura, dispersione |
| `climate.*` | l'arco nel tempo | costanti | vedi §8 |
| `sim_steps`, `step_stride`, `fps` | durata simulata vs durata vista | | vedi §8 |
| `guidance` (default **0**) | quanto gli agenti annusano l'errore | `0` | convergenza rapida ma si avvicina al barare — esposto per assaggiarlo, non per usarlo |

---

## 16. Decisioni prese

Le cinque domande aperte, chiuse.

| # | Domanda | Decisione |
| --- | --- | --- |
| 1 | Genoma | **Solo colore.** Il comportamento diventa §14.1, da provare a base tarata. |
| 2 | Clima | **Variabile, con la costante come caso particolare.** Stessa struttura `{start, end, easing}`; `start == end` ⇒ costante. Si decide sui test, non a tavolino. |
| 3 | Cristallizzazione | **Mai.** Processo idealmente infinito: vita del colore inversamente proporzionale all'errore, con `λ_min > 0` che non azzera mai il ricambio (§5.9). |
| 4 | Spazio colore | **OKLab**, per geni, distanze, blending e decadimento (§10). |
| 5 | `fitness_scale` | **Fedele** (`1`, errore a piena risoluzione di lavoro) come punto di partenza. |
| — | Propriocezione | **Aggiunta** dopo la revisione: run-and-tumble sul proprio `gain` (§5.2b). Costa due righe ed è l'unica cosa che rende efficiente il singolo agente invece della sola popolazione (§2.1). Legittima perché legge il tempo, non lo spazio (§2.2). |
| — | Cervello / neuroevoluzione | **Scartata** (§14.3): il limite non è la capacità di calcolo degli agenti ma quanto poco possono percepire, e percepire di più è barare. |
| — | Integrazione | **Isolata prima.** Laboratorio con immagine di test, bassa risoluzione, pochi agenti, GPU opzionale, prima di collegare gli stadi 1 e 2 (§11). |
| — | Durata | `sim_steps` e `fps` indipendenti: si simula lungo e si accelera in riproduzione (§8). |

---

## 17. Substrati scartati

Il tuo *"pixel o cellula o sezione"* ha tre letture; due sono state adottate insieme
(agente + deposito), una è stata scartata:

- **agente** — l'individuo si muove, mangia, muore, si riproduce. **Adottato**: è ciò che
  produce sciame, tracce e migrazione.
- **deposito / pixel** — la pittura ha una vita. **Adottato** come secondo livello (§6).
- **sezione / patch** — dividere il canvas in toppe, ognuna con un colore, e selezionare
  fra toppe. **Scartato**: è l'approssimazione genetica di immagini alla "Mona Lisa a
  poligoni". Converge bene, ma è statica: niente moto, niente sciame, niente pennellata.
  Perde tutto quello che ti piaceva delle particelle.
