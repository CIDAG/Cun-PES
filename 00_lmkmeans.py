# default libs
import sys
import os.path
import pickle
from configparser import ConfigParser
from glob import glob

# external libs
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances

# custom libs
import libs.xyz_tools as xyz

SIM_SIZE = 2501


def check_and_get_args() -> (
    list,
    str,
    str,
    bool,
    bool,
    str,
    float,
    float,
    str,
    float,
    list,
    str,
    str,
    list,
):
    """Checks if the program received the correct number of args.
    If it did, return a list of xyz files and an ndarray with
    extra information"""

    if len(sys.argv) < 2:
        print("Please inform settings.ini")
        exit()

    config = ConfigParser()
    config.read(sys.argv[1])

    temperature = config.getfloat("lm_only", "temperature")
    boltzmann = config.getfloat("lm_only", "boltzmann")
    clustering_method = config.get("lm_only", "method")
    use_extra_data = config.getboolean("everything", "use_extra_file")
    extra_file = config.get("lm_only", "extra_file_lm")
    xyz_path = config.get("lm_only", "xyz_path_lm")
    threshold = config.getfloat("lm_only", "similarity_threshold")
    outputs_folder = config.get("everything", "outputs_folder")
    randomseed = config.getint("everything", "randomseed")

    xyz_files = glob(xyz_path + "/*.xyz")
    xyz_files.sort()

    return_tuple = (
        xyz_files,
        xyz_path,
        extra_file,
        use_extra_data,
        clustering_method,
        temperature,
        boltzmann,
        threshold,
        outputs_folder,
        randomseed,
    )
    return return_tuple


def argsort(seq):
    return [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]


def load_extra_data(file_extra, xyz_files, use_extra_data):
    files_list = []
    energy_data = []
    extra_data = []

    with open(file_extra, "r") as f:
        _ = f.readline()
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
    pickle_path = outputs + "/matrixpickle.pickle"
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
    N = eigen_matrix.shape[1]

    # get formatted_extra_data concatenated to eigen_matrix
    if use_extra_data:
        eigen_matrix = np.concatenate(
            (np.array(eigen_matrix), extra_data), axis=1
        )

    eigen_matrix = StandardScaler().fit_transform(eigen_matrix)

    return N, eigen_matrix


def get_statistics(
    formatted_data: np.ndarray,
    energy_data: np.ndarray,
    labels: np.ndarray,
    params: list,
) -> (np.float64, np.ndarray, list):
    """Return the number of clusters, all-clusters silhouette, per-cluster
    silhouette, and also min, max, avg and variance values of extra data
    for each cluster"""

    n_clusters = params[0]

    # check if there are at least 2 clusters
    if n_clusters < 2:
        print("The clustering algorithm generated less than 2, end=' '")
        print("clusters (insufficient).")
        exit()

    # silhouette calculations
    all_clus_silhouette = metrics.silhouette_score(formatted_data, labels)
    per_sample_silhouette = metrics.silhouette_samples(formatted_data, labels)

    per_clus_silhouette = []
    per_clus_std_energy = []
    per_clus_var_energy = []
    per_clus_max_min = []

    # IDS = np.array([x for x in range(n_files)])

    for clus in range(n_clusters):
        per_clus_silhouette.append(
            np.max(per_sample_silhouette[labels == clus])
        )
        per_clus_std_energy.append(
            2.35482 * np.std(energy_data[labels == clus])
        )
        per_clus_var_energy.append(np.var(energy_data[labels == clus]))
        per_clus_max_min.append(
            np.max(energy_data[labels == clus])
            - np.min(energy_data[labels == clus])
        )

    clusters_statistics = (
        all_clus_silhouette,
        per_clus_silhouette,
        0,
        per_clus_std_energy,
        per_clus_var_energy,
        per_clus_max_min,
    )

    return clusters_statistics


def cluster_data(
    method: str,
    form_data: np.ndarray,
    energy_data: np.ndarray,
    params: list,
    randomseed: int,
) -> (np.ndarray, int, list):
    """cluster data with selected algorithm, then return several
    statistical measures"""

    # run chosen clustering method
    if method == "kmeans":

        cluster_alg = cluster.KMeans(
            n_clusters=params[0],
            n_init=params[1],
            max_iter=1000,
            random_state=randomseed,
        ).fit(form_data)
        n_clusters = params[0]

    else:
        print("Unavailable clustering method")
        exit()

    clusters_statistics = get_statistics(
        form_data, energy_data, cluster_alg.labels_, params
    )
    return cluster_alg.labels_, n_clusters, clusters_statistics


def clustering_analysis(
    N,
    clustering_method,
    formatted_data,
    energy_data,
    temperature,
    boltzmann,
    randomseed,
):
    maxClusters = int(np.ceil(np.sqrt(formatted_data.shape[0])))

    best_k = -1
    best_prop = 0
    best_labels = []
    best_res = []

    small_k = -1
    small_prop = 0
    small_labels = []
    small_res = []

    not_detected = True

    S = formatted_data.shape[0]

    target = 1.5 * N * boltzmann * temperature
    print("\nClusters with #%d atoms" % N)
    print("Number of samples: %d" % S)
    print("Temperature: %.1f" % temperature)
    print("Target 1.5NKT == %.3f" % target)
    print(
        "Evaluating with #clusters from 2 to %d [SQRT(%d)]" % (maxClusters, S)
    )

    for i in range(2, maxClusters + 1):
        print("\nSimulation with %d Clusters" % i)
        params = [i, 10, temperature, boltzmann]  # num_clus, n_init, T/K, B
        labels, n_clusters, res = cluster_data(
            clustering_method, formatted_data, energy_data, params, randomseed
        )
        silhouette_global = res[0]

        print("\tSilhouette: %.2f" % (silhouette_global))

        silhouette_clus = np.array(res[1])
        ids = np.where(silhouette_clus >= silhouette_global)[0]
        qtd_good_sil = len(ids)
        print("\tSilhouette max rate: %.2f" % (qtd_good_sil / i))

        std_energy = np.array(res[3])
        ids = np.where(std_energy > target)[0]
        qtd_bad_clus = len(ids)
        prop_std = (i - qtd_bad_clus) / i
        print(
            "\tNum of Clusters with std<=NKT: %d [out of %d] - Prop %.2f"
            % (i - qtd_bad_clus, i, prop_std)
        )

        var_energy = np.array(res[4])
        ids = np.where(var_energy > target)[0]
        qtd_bad_clus = len(ids)
        prop_var = (i - qtd_bad_clus) / i
        print(
            "\tNum of Clusters with var<=NKT: %d [out of %d] - Prop %.2f"
            % (i - qtd_bad_clus, i, prop_var)
        )

        max_min_energy = np.array(res[5])
        ids = np.where(max_min_energy > target)[0]
        qtd_bad_clus = len(ids)
        prop_max_min = (i - qtd_bad_clus) / i
        print(
            "\tNum of Clusters with max_min<=NKT: %d [out of %d] - Prop %.2f"
            % (i - qtd_bad_clus, i, prop_max_min)
        )

        counts = np.bincount(labels[labels >= 0])
        print("\tSize of each cluster:" + str(counts))

        if prop_max_min == 1 and not_detected:
            not_detected = False
            small_prop = prop_max_min
            small_k = i
            small_res = res
            small_labels = labels

        if prop_max_min > best_prop:
            best_prop = prop_max_min
            best_k = i
            best_res = res
            best_labels = labels

    res_small = (small_prop, small_k, small_res, small_labels)

    res_best = (best_prop, best_k, best_res, best_labels)
    return res_small, res_best


def get_representative(data, labels, n_clusters):

    representative = []
    for clus in range(n_clusters):
        # get elements that belong in the cluster
        elements_id = np.where(labels == clus)
        # get features of these elements
        cluster_elements = data[elements_id]
        # compute the center of mass
        centroid = np.average(cluster_elements, axis=0)

        # get the closest element to the centroid
        min_distance = sys.maxsize
        min_element = -1
        for elem in elements_id[0]:
            actual_distance = np.linalg.norm(data[elem] - centroid)
            if actual_distance < min_distance:
                min_distance = actual_distance
                min_element = elem
        representative.append(min_element)

    return representative


def print_representatives(
    formatted_data, files_list, method, res_small, res_best, outputs
):

    # Best
    num_clus = res_best[1]
    labels = res_best[3]
    sil = res_best[2][0]
    prop = res_best[0]

    print("Best #clusters in range: %d" % num_clus)
    with open(f"{outputs}/best_output.txt", "w") as out:
        print(f"Clustering Method: {method}", file=out)
        print(f"#Clusters {num_clus}", file=out)
        print(f"Silhouette {sil}", file=out)
        print(f"max_min <= NKT Proportion: {prop}", file=out)
        print("\n .... \n", file=out)

        representatives = get_representative(formatted_data, labels, num_clus)
        for i, rep in zip(range(num_clus), representatives):
            print(
                "Representative for Cluster %d - %s" % (i + 1, files_list[rep])
            )
            print(
                f"Representative for Cluster {i+1} - {files_list[rep]}",
                file=out,
            )
            print(files_list[labels == i], file=out)

    if res_small[1] > 0:
        # Smallest (NKV ok)
        num_clus = res_small[1]
        labels = res_small[3]
        sil = res_small[2][0]
        prop = res_small[0]
        print("Optimal #clusters from energy: %d " % num_clus)
        with open(f"{outputs}/small_output.txt", "w") as out:
            print(f"Clustering Method: {method}", file=out)
            print(f"#Clusters {num_clus}", file=out)
            print(f"Silhouette {sil}", file=out)
            print(f"max_min <= NKT Proportion: {prop}", file=out)
            print("\n .... \n", file=out)

            representatives = get_representative(
                formatted_data, labels, num_clus
            )
            for i, rep in zip(range(num_clus), representatives):
                print(
                    "Representative for Cluster %d - %s"
                    % (i + 1, files_list[rep])
                )
                print(
                    f"Representative for Cluster {i+1} - {files_list[rep]}",
                    file=out,
                )
                print(files_list[labels == i], file=out)
    else:
        print("\nFail to achieve max_min < NKT for all clusters\n\n")


def exclude_similar(
    files_list, extra_data, formatted_data, energy_data, threshold
):
    dist = euclidean_distances(formatted_data, formatted_data)

    Samples = formatted_data.shape[0]

    ids = np.where(dist == 0)
    dist[ids] = np.max(dist)

    ids = np.where(dist < threshold)
    dist[ids] = 0

    ids = []
    for i in range(Samples):
        line = dist[i, i + 1 :]
        if len(np.where(line == 0)[0]) == 0:
            ids.append(i)

    files_list = np.array(files_list)[ids]
    if extra_data:
        extra_data = extra_data[ids, :]
    formatted_data = formatted_data[ids, :]
    energy_data = energy_data[ids]

    return files_list, extra_data, formatted_data, energy_data


def pickle_all(
    formatted_data, energy_data, res_small, res_best, xyz_files, outputs
):
    """Pickle important information for the next step"""
    with open(f"{outputs}/lm_pickles.pickle", "wb") as pfile:
        pickle.dump(formatted_data, pfile)
        pickle.dump(energy_data, pfile)
        pickle.dump(res_small, pfile)
        pickle.dump(res_best, pfile)
        pickle.dump(len(xyz_files), pfile)


def main():
    print("Loading and formatting data...")
    (
        xyz_files,
        xyz_path,
        extra_file,
        use_extra_data,
        clustering_method,
        temperature,
        boltzmann,
        threshold,
        outputs_folder,
        randomseed,
    ) = check_and_get_args()

    files_list, energy_data, extra_data = load_extra_data(
        extra_file, xyz_files, use_extra_data
    )
    print(f"Input file <{extra_file}>")
    N, formatted_data = format_data(
        xyz_files, extra_data, use_extra_data, outputs_folder
    )

    print("Eliminating similar samples with threshold %.3f" % threshold)
    files_list, extra_data, formatted_data, energy_data = exclude_similar(
        files_list, extra_data, formatted_data, energy_data, threshold
    )

    # LOOP Clustering
    print("Running clustering analysis...")
    res_small, res_best = clustering_analysis(
        N,
        clustering_method,
        formatted_data,
        energy_data,
        temperature,
        boltzmann,
        randomseed,
    )

    print("Saving results...")
    print_representatives(
        formatted_data,
        files_list,
        clustering_method,
        res_small,
        res_best,
        outputs_folder,
    )

    pickle_all(
        formatted_data,
        energy_data,
        res_small,
        res_best,
        xyz_files,
        outputs_folder,
    )

    print("Done!")


if __name__ == "__main__":
    main()
