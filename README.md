## masterthesis 

# Evaluation of Anomaly Detection in Fuzzy Hashes through Synthetic Data

Prerequisites:

Ubuntu 21.04 sollte alles stemmen können. Damit die executables für die fuzzy hashes laufen sollte eigentlich keinen weiteren Pakete nötig sein. Falls irgendwas mit den fuzzy hashes nicht klappt einfach schnell schreiben. Ansonsten eigentlich alles in Python programmiert. Folgenden libraries sind per pip3 zu installieren: 

<
pytorch
tabnanny
pickle 
matplotlib
pandas
matplotlib
termplotlib
>

### Wie man die testdaten aquieriert: 

Für das Training habe ich folgdende Daten corproa benutzt: 

- govdocs 1: das skript 'govdocs_downloader.py' kann genutzt werden um den gesamten corpus herunterzuladen. In Zeile 7 in range() Lässt sich festlegen wieviel heruntergeladen werden soll (0-700).
- napierone corpus geht über das [git repository](https://github.com/simonrdavies/NapierOne)
- Link zum Fuse corpus (xlsx) gibts in diesem [Paper](https://www.researchgate.net/publication/308861425_Fuse_A_Reproducible_Extendable_Internet-Scale_Corpus_of_Spreadsheets)
- Link zum javascript corpus von SRI labs gibts [hier](https://www.sri.inf.ethz.ch/js150)
--

### Wie man trainingsdaten erstellt: 

Über das `main.py` skript lässt sich ein datenset erstellen. Folgende Flags sind zu setzen: 

DATASET_FILETYPE: 
- `all` für gemischte files
- `js` für javascript
- `pdf`  für pdf
- `random` für random generierte files
- ...

MAX_FRAGMENT_PERCENTAGE: die maximale größe der Anomalie
MIN_FRAGMENT_PERCENTAGE: die minimale Größe der Anomalie
SAMPLE_FILES_PATH: hier den pfad angeben aus dem die original files genommen werden sollen.

IN Zeile 295, 298, 301, 303 stehen jeweils die pfade wohin die trainignsdaten dann gesaved werden. Einmal für ssdeep und einmal für tlsh. 
jeder datensatz pro hash besteht aus einem `anomaly` und einem `normal`. In Zeile 293 am Ende ist einmal einzutragen wohin die anomalie gesichert werden soll. 
IMMER IN DENSELBEN ORDNER WIE AUCH DIE ANDEREN TESTDATENSÄTZE. sie wird für die testerstellung gebraucht. 

--
### Wie man ein model trainiert: 

Die beiden skripte `feedforward.py`  und `transformer.py` trainiern die model. In Zeile 193 ist anzugeben welcher fuzzy hash benutzt wird und in Zeile 195 und 196 ist anzugeben welche trainingsdaten zu verwenden sind (pfade). In Zeile 199  ist einzutragen wie groß der datensatz sein soll auf dem trainert wird.
Ich hab die beiden skripte immer in visual studio code geöffnet. Wenn man dort bis zur Zeile 605 scrollt und per rechtsklick: 'Run to line in interactive Window' angibt. Dann öffnet sich ein Fenster in dem man die loss kurve geplottet kriegt etc. In Zeile 612 ist der speicherort des models anzugeben. 

Nach dem trainieren des models werden die max_hash_len, die hidden_size und die vocab_size ausgegeben. Wenn man das model klasssifizieren lassen will sollte man sich diese Parameter aufschreiben. 
--
### Wie man model testet.

Per filtype gibt es eine sogenannte `testconfig.yml` in den ordnern ´evaluation_testcase_*filetype*´ *pdf, xlsx, random etc. In dieser config kann man angeben. Diese Configs erlauben es die evaluationskonfigurationen der testdaten zu ändern. Die wichtigsten (alle anderen sollte selbsterklärend sein): 

- Fragmen_size: hierüber könen mindest und maximalegröße der anomalie in den files angegeben werden. 
- Fuzzy_hashing_algorithm: einfach einkommentieren welcher fuzzy hash ausprobiert werden soll. 
- Model_Evaluation: Es wird immer genau das Model ausgeführt das genau eine `Model_Evaluation` die restlichen Modelle auskommentieren. 

Schlieslich um den test durchzuführen: `´python3 evaluation_with_fuzzy_hashes.py *pfad zur entsprechenden testconfig.yml*´ 
Das Ergebniss wird für jedes Model visualisiert in einer pdf datei in einem angegeben Ordner gespeichert. 

