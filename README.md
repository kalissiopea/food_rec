# food_rec
Questa cartella contiene tutti i passaggi che sono stati fatti nel progetto:

+ divide.py : divide il dataset in modo che ci siano 500 img per categoria per il training, 250 img per categoria per il validation e testing, funziona in locale
+ preprocess.ipynb : contiene un esempio delle trasformazioni che si applicano ai dati di training
+ cvprojectforsefinal.ipynb : contiene la parte di preprocessing e augmentation dei dati, il caricamento dei dati, il training e il validation del modello resnet50, il plot di loss e accuracy e la stampa della matrice di confusione ottenuta validando la parte di test del dataset
+ app : la cartella app contiene l'app web su cui si valida il modello per predire gli ingredienti insieme a static e template che sono lo scheletro dell'app web
