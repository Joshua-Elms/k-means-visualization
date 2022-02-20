import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from math import dist
import imageio
import warnings


def generate_data(k, half_of_points, dims, distr_means, stdevs):
    """
    Uses random generator from NumPy to create two clusters for us to play with
    Based on the min and max of each cluster, determine the initial conditions for our centroids

    Args:
        k: num of centroids/clusters to be found
        half_of_points: Multiply this by 2 to find the number of clusterable data points from which we will make a data frame, 1/2 for each cluster
        dims: How many dimensions the data will span (each will be a column in final df); separate from clusters, as this will only generate 3 clusters
        distr_means: Where the clusters will be centered around (mean for normal distr.)
        stdevs: Measure of spread for each cluster (stdev for normal distr.)

    Returns:
        init_data_matrix: nd.array containing our points for clustering
        means: initial values for each of the k centroids, "dim" dimensional random values from ranges of init_data_matrix
    """
    rng = np.random.default_rng()
    clust1 = rng.normal(loc=distr_means[0], scale=stdevs[0], size=(half_of_points, dims))
    clust2 = rng.normal(loc=distr_means[1], scale=stdevs[1], size=(half_of_points, dims))

    data = np.concatenate((clust1, clust2)) # create nd.array from data

    columns = [data[:, i] for i in range(dims)] # get list of all columns

    dim_ranges = [(column.min(), column.max()) for column in columns] # get min and max of each column into tuple

    means = [rng.uniform(*dim_range, size=k).tolist() for dim_range in dim_ranges] # create dim sized, randomly selected points to represent centroids

    return data, means


def get_column_major(arr): 
    """
    Internal function for switching nested list from row major to column major
    """
    cm_arr = []
    for c in range(len(arr[0])):
        new_col = []
        for row in arr:
            new_col.append(row[c])
        cm_arr.append(new_col)

    return cm_arr


def build_df(matrix, rm_means, dims, k_num): 
    """
    Create dataframe containing all points, centroids, and type / cluster labels for each

    Args:
        matrix: matrix containing points to cluster
        rm_means: k_num points (repr'ed by tpls) that fall randomly in the range of each dimension
        dims: int, # of numeric dimensions for data
        k_num: # of cluster to form

    Returns:
        df: Dataframe containing all necessary data for clustering
    """

    # Initialize df containing only generated points for clustering
    df1 = pd.DataFrame(matrix)
    col_names = ("X", "Y")
    df1.columns = col_names # set names for all of our points dimensions
    df1["Type"] = "data" # Type will be either data or centroid
    df1["Cluster"] = "None" # Cluster will be one of the centroids (k1, k2, ...)

    # Initialize df containing all centroids
    cm_means = get_column_major(rm_means)
    k_dict = {col_names[d]:rm_means[d] for d in range(dims) for i in range(k_num)} # Maps Dim1: initial dim1 values for each k, etc
    k_dict["Type"] = "centroid"
    k_dict["Cluster"] =  [f"C{i+1}" for i in range(k_num)] # Assigns a cluster to each centroid

    # Concatenate the two df's above
    df_k = pd.DataFrame(k_dict)
    df = pd.concat([df1, df_k], ignore_index=True)
    df.index += 1

    return df


def parse_data(path, k_num, delim=","): 
    """
    Generates dataframe using user data at a given path instead of random data

    Args:
        path: str path to 2 dimensional dataset
        k_num: number of clusters to group into

    Returns: 
        data: row major matrix of data
        d_means: randomly selected points from inside data ranges
    """
    data = np.genfromtxt(path, delimiter=delim)
    columns = [data[:, i] for i in range(2)] # get list of all columns
    dim_ranges = [(column.min(), column.max()) for column in columns] # get min and max of each column into tuple
    rng = np.random.default_rng()
    d_means = [rng.uniform(*dim_range, size=k_num).tolist() for dim_range in dim_ranges] # create dim sized, randomly selected points to represent centroids

    return data, d_means


def Assignment(df_in, k_num):
    """
    Given a dataframe containing points to be clustered and some k centroids,
    assign each point to the nearest (euc. distance) centroid's cluster using math.dist()

    Args:
        df_in: dataframe

    Returns: 
        df_out: reassigned points to new dataframe
        same: Bool, whether or not df_in == df_out
    """
    centroids = df_in[df_in.Type == "centroid"]
    centroids.index = tuple(range(k_num))

    points = df_in[df_in.Type != "centroid"]

    df_point_dists = [points.copy(deep=True) for k in range(k_num)] # must be able to calculate distance from points for each centroid

    for i in range(k_num): # Calculate distance between some k and some point for each k and point
        df_point_dists[i]["Dist"] = 0
        for index, row in df_point_dists[i].iterrows():
            df_point_dists[i].iloc[index-1, 4] = dist((row["X"], row["Y"]), (centroids.iloc[i]["X"], centroids.iloc[i]["Y"]))

    df_out = df_in.copy(deep=True) # create output df

    for i in range(len(points)): # for each set of distances, choose the minimum one and assign the cluster for the point it was closest to
        distances = [df_point_dists[v].iloc[i]["Dist"] for v in range(k_num)]
        min_index = distances.index(min(distances))
        df_out.iloc[i, 3] = centroids.iloc[min_index]["Cluster"]

    return df_out, df_in


def Update(df, k_num): 
    """
    Given a df with clusters assigned, change centroids to reflect the actual centers of their clusters
    
    Args:
        df

    Returns: 
        df
    """
    l1 = list(df[df["Type"] == "centroid"]["X"])
    l2 = list(df[df["Type"] == "centroid"]["Y"])

    k_origs = get_column_major(((l1), (l2)))

    X_mean = [df.iloc[1:-k_num][df["Cluster"] == f"C{i+1}"]["X"].mean() for i in range(k_num)]
    Y_mean = [df.iloc[1:-k_num][df["Cluster"] == f"C{i+1}"]["Y"].mean() for i in range(k_num)]

    seq = tuple(range(-1, -k_num-1, -1))

    for i in range(k_num): # (i - 1)
        df.iloc[seq[i], 0] = X_mean[seq[i]]
        df.iloc[seq[i], 1] = Y_mean[seq[i]]

    l3 = list(df[df["Type"] == "centroid"]["X"])
    l4 = list(df[df["Type"] == "centroid"]["Y"])

    k_modded = get_column_major(((l3), (l4)))
     
    largest_move = max([dist((k_origs[i][0], k_origs[i][1]), (k_modded[i][0], k_modded[i][1])) for i in range(k_num)])

    return df, largest_move 


def generate_graph(df, save_path, step, k_num, plines=None, plabels=None):
    """
    Generate plot of all points and centroids

    Args: 
        df: data to be graphed, must have "X", "Y", "Cluster", and "Type" attrs
        save_path: relative / absolute path to folder where plot should be saved
        step: which iteration of the algorithm you are on 
    """ 

    all_color_dict = {
    "None":"#000000", "C1":"#FF0000", "C2":"#0000FF",
    "C3":"#00FF00", "C4":"#D030C9", "C5":"#D09D30", 
    "C6":"#8a4ead", "C7":"#8ab5a4", "C8":"#4a0909"
    }

    if step > 0:

        sns.set_theme()
        sp = sns.scatterplot(data=df, x="X", y="Y", hue="Cluster", palette=all_color_dict, style="Type", size="Type", sizes=[25,80]).set(title=f"K-Means Algorithm: Step {step}")
        plt.legend([],[], frameon=False)
        sp[0].figure.legend(plines, plabels, bbox_to_anchor=(0.915, 0.88), loc='upper left', borderaxespad=0)
        plt.ylabel("Y", rotation=0)

        plt.savefig(f"{save_path}/kmeans_step{step}.jpeg", bbox_inches='tight')
        plt.clf()
        pass 

    else:
        sns.set_theme()
        sp = sns.scatterplot(data=df, x="X", y="Y", hue="Cluster", palette=all_color_dict, style="Type", size="Type", sizes=[25,80]).set(title=f"K-Means Algorithm: Step {step}")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.ylabel("Y", rotation=0)
        plt.savefig(f"{save_path}/kmeans_step{step}.jpeg", bbox_inches='tight')
        
        lines_labels = [ax.get_legend_handles_labels() for ax in sp[0].figure.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        plt.clf()
        return lines, labels


def controller(df, k_num, path):
    """
    Puts data through steps and calls other function until it is determined that the clusters are complete
    """
    changing = True
    step = 0
    lines, labels = generate_graph(df, path, step, k_num)
    while changing:
        df, compare = Assignment(df, k_num)
        df, change = Update(df, k_num)
        changing = False if change < 0.1 else True
        step += 1
        generate_graph(df, path, step, k_num, lines, labels)
        print(f"Iteration {step} complete")

    return step


def make_gif(path_in, iters): 
    """
    Uses imageio library to convert a series of images into a gif
    
    Args: 
        path_in: str filepath for folder storing jpegs
    """
    filenames = [f"{path_in}/kmeans_step{i}.jpeg" for i in range(iters)]
    movie_path = f"{path_in}/movie.gif"

    with imageio.get_writer(movie_path, mode='I', duration=3) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    pass


def set_params(k=2, half_of_points=20, dims=2, means=(5, 15), stdevs=(1, 1), image_folder_path="", data_path=""): 
    """
    Determine the parameters of the algorithm

    Args:
        None

    Returns: 
        k: int, determines # of centroids that will be chosen (final # of clusters)
    """
    means = (5, 15)
    stdevs = (2, 2)
    half_of_points = 30
    k = 3
    image_folder_path = "/Users/joshuaelms/Desktop/github_repos/k-means-visualization/k-means-visualization/images"
    data_path = "http://cs.joensuu.fi/sipu/datasets/unbalance.txt"
    delim = None
    return k, half_of_points, dims, means, stdevs, image_folder_path, data_path, delim


def main():
    warnings.filterwarnings("ignore")
    k_num, half_of_points, dims, distr_means, stdevs, image_folder_path, data_path, delim = set_params()
    data, means = parse_data(data_path, k_num, delim) if data_path else generate_data(k_num, half_of_points, dims, distr_means, stdevs)
    df = build_df(data, means, dims, k_num)
    iters = controller(df, k_num, image_folder_path)
    make_gif(image_folder_path, iters)
    print("Visualization complete")


if __name__ == "__main__": 
    main()
