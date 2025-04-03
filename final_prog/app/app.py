import os
from flask import Flask, render_template, redirect, request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def remove_hidden_files(directory):
    # Scorri in maniera ricorsiva all'interno della directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Verifica se il nome del file inizia con un punto
            if file.startswith('.'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")


app = Flask(__name__)

# Sostituisci con il percorso della directory da cui vuoi rimuovere i file nascosti
directory_path = '/Users/lunaticyber/Desktop/dsToProcess'
remove_hidden_files(directory_path)

# Percorso alla cartella con le immagini di test
food101 = "/Users/lunaticyber/Desktop/dsToProcess/test"
foods = sorted(os.listdir(food101))

# Carica il modello ResNet50 e i pesi addestrati
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(model.fc.in_features, 20)
)

model.load_state_dict(torch.load('resnet50modelforsefinal.pth', map_location=device))
model.eval()
model.to(device)
print('Modello caricato correttamente!')

# Trasformazioni per le immagini (adatta le dimensioni e normalizza)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/NotFile')
def no_file():
    return render_template('indexNoFile.html')


@app.route('/MoreFile')
def more_file():
    return render_template('indexMoreFile.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return redirect('/NotFile')

    file_path = os.path.join('./upload', file.filename)
    file.save(file_path)

    return redirect("/MoreFile")


@app.route('/give', methods=['GET'])
def give_recipe():
    labels = []
    images_dir = './upload/'
    for img_name in os.listdir(images_dir):
        if not img_name.startswith('.'):
            # Carica l'immagine e applica le trasformazioni
            img_path = os.path.join(images_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0)
            image = image.to(device)

            # Predici con il modello
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                pred_value = foods[predicted.item()]

            # Aggiungi l'etichetta predetta alla lista
            labels.append(pred_value)

    # Crea la stringa di ricerca per Google
    string = "Recipes+with+" + "+".join(labels)
    query = request.args.get('q', default=string)
    google_search_url = f"https://duckduckgo.com/?q={query}&ia=web"
    return redirect(google_search_url)


if __name__ == '__main__':
    app.run(debug=True)
