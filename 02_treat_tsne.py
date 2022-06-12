import sys
from configparser import ConfigParser
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Please inform settings.ini")

config = ConfigParser()
config.read(sys.argv[1])
outputs = config.get("everything", "outputs_folder")

knn_n_neighbors = 5
data = open(f"{outputs}/tsne_xyz.dat", "r").readlines()
x = []
y = []
z = []

for line in data:
    splited = line.split(" ")
    x.append(float(splited[0]))
    y.append(float(splited[1]))
    z.append(float(splited[2]))

pair_data = []
for i in range(0, len(x)):
    pair_data.append([x[i], y[i]])

pair_data = np.array(pair_data)

clustering = DBSCAN(eps=7, min_samples=250, algorithm="auto").fit(pair_data)
lbl = clustering.labels_

cont = [0]
emin = [+9999.0]
emax = [-9999.0]
imin = [0]
for i in range(max(lbl)):
    cont.append(int(0))
    imin.append(int(0))
    emin.append(+9999.0)
    emax.append(-9999.0)

for i in range(len(lbl)):
    if lbl[i] > -1:
        cont[lbl[i]] = cont[lbl[i]] + 1
        if emin[lbl[i]] > z[i]:
            emin[lbl[i]] = z[i]
            imin[lbl[i]] = i
        if emax[lbl[i]] < z[i]:
            emax[lbl[i]] = z[i]

# KNN =========================================================================
train_data = []
train_labels = []
predict_data = []


for i in range(len(lbl)):
    if lbl[i] != -1:
        train_data.append(pair_data[i])
        train_labels.append(lbl[i])
    else:
        predict_data.append(pair_data[i])


knn_model = KNeighborsClassifier(n_neighbors=knn_n_neighbors)
knn_model.fit(train_data, train_labels)

predicted = list(knn_model.predict(predict_data))

final_labels = [i if i != -1 else predicted.pop(0) for i in lbl]

with open(f"{outputs}/tsne_tratado.dat", "w+") as file:
    for i, v in enumerate(final_labels):
        file.write(f"{i} {v}\n")

# =============================================================================

fig, ax = plt.subplots()
scatter = ax.scatter(x, y, c=final_labels, cmap=plt.cm.jet, marker=".")
ax.set_title("Clusters in t-SNE final data - Post KNN")
ax.grid(True)
# produce a legend with the unique colors from the scatter
legend1 = ax.legend(
    *scatter.legend_elements(), loc="lower left", title="Clusters"
)
ax.add_artist(legend1)
ax.set_xlabel("Comp. 1")
ax.set_ylabel("Comp. 2")
fig.tight_layout()
plt.savefig(f"{outputs}/tsne_tratado.png", dpi=400)
