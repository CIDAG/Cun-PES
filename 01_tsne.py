# default libs
import sys
import os.path
import pickle
from configparser import ConfigParser
from glob import glob

# external libs
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# custom libs
import libs.xyz_tools as xyz

SIM_SIZE = 2501


def argsort(seq):
    return [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]


def get_config():
    """Check for settings.ini file and setup values"""

    if len(sys.argv) < 2:
        print("Please inform settings.ini")
        exit()

    config = ConfigParser()
    config.read(sys.argv[1])

    extra_file_md = config.get("tsne_forwards", "extra_file_md")
    xyz_path_md = config.get("tsne_forwards", "xyz_path_md")
    outputs_folder = config.get("everything", "outputs_folder")
    use_extra_data = config.get("everything", "use_extra_file")
    randomseed = config.getint("everything", "randomseed")

    xyz_files_md = glob(xyz_path_md + "/*.xyz")
    xyz_files_md.sort()

    # get all MD names
    name_set = set()
    for file in xyz_files_md:
        tmp_name = file.split("/")[-1]
        tmp_name = tmp_name.split("s")[0]
        name_set.add(tmp_name)
    name_set = sorted(name_set)

    return (
        extra_file_md,
        xyz_files_md,
        outputs_folder,
        name_set,
        use_extra_data,
        randomseed,
    )


def load_extra_data(file_extra, xyz_files, use_extra_data):
    files_list = []
    energy_data = []
    extra_data = []

    with open(file_extra, "r") as f:
        _ = f.readline()  # cabeçalho
        for line in f:
            lin = line.split()
            files_list.append(lin[0])
            energy_data.append(lin[1])
            if use_extra_data:
                extra_data.append(lin[2:])

    files_list = np.array(files_list)
    energy_data = np.array(energy_data).astype(float)
    extra_data = np.array(extra_data).astype(float)

    index = argsort(files_list)
    files_list = files_list[index]
    energy_data = energy_data[index]
    print(energy_data.shape[0])
    energy_data = energy_data.reshape((energy_data.shape[0], 1))

    if use_extra_data:
        extra_data = extra_data[index]

    for f1, f2 in zip(files_list, xyz_files):
        if f1 not in f2:
            print("\nXYZ files do not match with the extra data\n\n")
            exit()

    return files_list, energy_data, extra_data


def format_data(
    xyz_files: list, extra_data: np.ndarray, use_extra_data: bool, outputs: str
) -> (np.ndarray):
    """Format the data so it's ready to go thought a clustering process. The
    return will be a matrix with each line containing eigen values and any
    other extra data given by the user. The number of columns of special data
    is also returned."""

    # if there's a pickled version of eigenmatrix, use it.
    pickle_path = outputs + "/matrixpickle_md.pickle"
    if os.path.isfile(pickle_path):
        print("Using pickled data. Be sure it's up to date!")
        with open(pickle_path, "rb") as picklefile:
            eigen_matrix = pickle.load(picklefile)
    else:
        print("Making calculations and pickle...")
        # create a matrix that contains the eigen values for each element
        eigen_matrix = []
        for file in xyz_files:
            print(f"Calculating for {file}")
            num_of_atoms, atom_types, coords = xyz.read(file)
            aux_m = xyz.eigen_coulomb(num_of_atoms, atom_types, coords)
            eigen_matrix.append(aux_m)
        # save eigen matrix to a pickle file
        with open(pickle_path, "wb") as picklefile:
            pickle.dump(eigen_matrix, picklefile)

    eigen_matrix = np.array(eigen_matrix)

    # get formatted_extra_data concatenated to eigen_matrix
    if use_extra_data:
        eigen_matrix = np.concatenate(
            (np.array(eigen_matrix), extra_data), axis=1
        )

    eigen_matrix = StandardScaler().fit_transform(eigen_matrix)

    return eigen_matrix


def plotTSNE(
    formatted_data,
    energy_data,
    res_small,
    res_best,
    formatted_data_md,
    energy_data_md,
    name_set,
    n_files,
    outputs,
    randomseed,
    extra_file_md,
):
    # get points to be plotted
    idset = []
    with open(extra_file_md) as f:
        for i, line in enumerate(f):
            lsplit = line.split()[0].split(".")[0].split("s")
            if lsplit[0] in name_set[1:] and lsplit[1] == "0001":
                idset.append(i - 1)

    tsne = TSNE(n_components=2, init="random", random_state=randomseed)
    # reduced_data = tsne.fit_transform(formatted_data)
    print("starting t-sne")

    reduced_data_md = tsne.fit_transform(formatted_data_md)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(
        reduced_data_md[n_files : len(reduced_data_md), 0],
        reduced_data_md[n_files : len(reduced_data_md), 1],
        c=energy_data_md[n_files : len(reduced_data_md), 0],
        marker=".",
    )
    lmx = [reduced_data_md[id_item, 0] for id_item in idset]
    lmy = [reduced_data_md[id_item, 1] for id_item in idset]
    ax.scatter(lmx, lmy, c="r", marker="x")
    ax.set_title("t-SNE with MD Results")
    ax.grid(True)
    ax.set_xlabel("Comp. 1")
    ax.set_ylabel("Comp. 2")
    fig.tight_layout()
    plt.savefig(f"{outputs}/BestTSNE.png", dpi=400)

    x = reduced_data_md[n_files : len(reduced_data_md), 0]
    y = reduced_data_md[n_files : len(reduced_data_md), 1]
    z = energy_data_md[n_files : len(reduced_data_md), 0]

    with open(f"{outputs}/name_set.dat", "w+") as file:
        for i in range(len(name_set)):
            if i > 0:
                file.write(f"{list(name_set)[i]}\n")

    with open(f"{outputs}/tsne_xyz.dat", "w+") as file:
        for i in range(len(x)):
            file.write(f"{x[i]} {y[i]} {z[i]}\n")

    deltaX = (max(x) - min(x)) / 20
    deltaY = (max(y) - min(y)) / 20
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(lmx, lmy, c="r", marker="x")
    cbar = plt.colorbar()  # draw colorbar
    cbar.set_label("Energy (kcal/mol)", labelpad=5)
    ax.set_xlabel("Comp. 1")
    ax.set_ylabel("Comp. 2")
    plt.title("Potential Energy Surface projected using t-SNE")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.savefig(f"{outputs}/pot_en_suf_tsne.png", dpi=400)

    x = reduced_data_md[n_files : len(reduced_data_md), 0]
    y = reduced_data_md[n_files : len(reduced_data_md), 1]
    z = energy_data_md[n_files : len(reduced_data_md), 0]
    step = 50

    xt = []
    yt = []
    for i in range(len(name_set) - 1):
        xt.append(
            [
                reduced_data_md[j, 0]
                for j in range(
                    n_files + i * SIM_SIZE, n_files + (i + 1) * SIM_SIZE, step
                )
            ]
        )
        yt.append(
            [
                reduced_data_md[j, 1]
                for j in range(
                    n_files + i * SIM_SIZE, n_files + (i + 1) * SIM_SIZE, step
                )
            ]
        )

    deltaX = (max(x) - min(x)) / 20
    deltaY = (max(y) - min(y)) / 20
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    fig = plt.figure()
    ax = fig.add_subplot()

    for i, v in enumerate(name_set):
        if i > 0:
            ax.plot(xt[i - 1], yt[i - 1], "o--", label=v)
    plt.legend()

    ax.scatter(lmx, lmy, c="r", marker="x")
    cbar = plt.colorbar()  # draw colorbar
    cbar.set_label("Energy (kcal/mol)", labelpad=5)
    ax.set_xlabel("Comp. 1")
    ax.set_ylabel("Comp. 2")
    plt.title("Potential Energy Surface projected using t-SNE")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.savefig(f"{outputs}/pot_en_suf_tsne2.png", dpi=400)

    x = reduced_data_md[n_files : len(reduced_data_md), 0]
    y = reduced_data_md[n_files : len(reduced_data_md), 1]
    z = energy_data_md[n_files : len(reduced_data_md), 0]

    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.imshow(np.rot90(f), cmap="coolwarm", extent=[xmin, xmax, ymin, ymax])
    cset = ax.contour(xx, yy, f, colors="k")
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel("Comp. 1")
    ax.set_ylabel("Comp. 2")
    plt.title("2D density of samples estimation - t-SNE")
    plt.savefig(f"{outputs}/2d_density.png", dpi=400)


def load_pickle(outputs):
    """Unpickle important information from the previous step"""
    with open(f"{outputs}/lm_pickles.pickle", "rb") as pfile:
        formatted_data = pickle.load(pfile)
        energy_data = pickle.load(pfile)
        res_small = pickle.load(pfile)
        res_best = pickle.load(pfile)
        len_xyz_files = pickle.load(pfile)

    return (formatted_data, energy_data, res_small, res_best, len_xyz_files)


def main():

    (
        extra_file_md,
        xyz_files_md,
        outputs_folder,
        name_set,
        use_extra_data,
        randomseed,
    ) = get_config()

    (
        formatted_data,
        energy_data,
        res_small,
        res_best,
        len_xyz_files,
    ) = load_pickle(outputs_folder)

    (files_list_md, energy_data_md, extra_data_md) = load_extra_data(
        extra_file_md, xyz_files_md, use_extra_data
    )

    print("Plotting TSNE")
    formatted_data_md = format_data(
        xyz_files_md, extra_data_md, use_extra_data, outputs_folder
    )

    plotTSNE(
        formatted_data,
        energy_data,
        res_small,
        res_best,
        formatted_data_md,
        energy_data_md,
        name_set,
        len_xyz_files,
        outputs_folder,
        randomseed,
        extra_file_md,
    )

    print("Done!")


if __name__ == "__main__":
    main()
