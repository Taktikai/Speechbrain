import os
import pickle
import torchaudio
from speechbrain.pretrained import EncoderClassifier


classifier = EncoderClassifier.from_hparams(source=r"speechbrain/spkrec-ecapa-voxceleb",
                                            savedir=r"C:\Users\La Bouff Alexander\Desktop\pythonProject\ecapa")
directory = r"C:\Users\La Bouff Alexander\Desktop\pythonProject\patologias_adatbazis\hangok_16kHz"

x = []
a = 0
y = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    if os.path.isfile(f):
        signal, fs = torchaudio.load(f)
        embeddings = classifier.encode_batch(signal)
        output_probs, score, index, text_lab = classifier.classify_batch(signal)
        x.append([output_probs + score + index])
        a = a + 1
        print(a, "/457")

        if filename[0] == "H":
            y.append(0)
        else:
            y.append(1)

with open(r"C:\Users\La Bouff Alexander\Desktop\pythonProject\ecapa\ecapa.pickle", "wb") as d:
    pickle.dump(x, d)

with open(r"C:\Users\La Bouff Alexander\Desktop\pythonProject\ecapa\ecapaname.pickle", "wb") as e:
    pickle.dump(y, e)

print("x ", len(x))
print("y ", len(y))
print("y ", y)
