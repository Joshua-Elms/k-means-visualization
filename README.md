# k-means-visualization
Generates iterative visualization (gif) of 2-dimensional data with k-means clustering algorithm.

How to Use This Program:
  The majority of changes you will make take place in the set_params() function.
  Importantly, you must assign the image_folder_path to a folder on your system, or you will not see output for the program.
  By changing the variables in the body of the function, you can either randomly generate two gaussian clusters according to your own distribution,
  or you can pass a path to your own dataset, which must be 2-dimensional and entirely numeric (example below).
  If you provide a path to your own dataset, the data within will be used; if you assign the data_path None or "" then random data will be generated.


Example data.csv:
0,1
0,2
0,3
1,10
1,20
1,30

Libraries: 
  matplotlib - plotting
  seaborn - plotting
  numpy - data manipulation
  pandas - data manipulation
  math - euclidean distance calculation
  imageio - generate gif from jpegs
  warnings - remove unnecessary warnings

This project has three main components: 
  1. Data Generation
  2. Algorithm Implementation
  3. Visualization

I will explore each of these in more detail, along with the functions that are included in them.

1. Data Generation

  Three primary functions make up this section: parse_data(), generate_data(), and build_df().
  Only one of the first two will be called, depending on how the user intends to use the program.
  If the user leaves the data_path variable in set_params() blank, then the data will be generated according to parameters shown using generate_data().
  If the user provides a path to a 2-dimensional, numeric dataset with no header then that data will be parsed and used for clustering instead.
  For custom data, any tabular format is acceptable but default delimiter will be "," unless changed in set_params().
  Whatever data is generated is then passed on to build_df(), which creates the dataframe that is used for the rest of the program's execution.
  
2. Algorithm Implementation

  This section is much easier to understand after learning how k-means works, so check out the wikipedia: https://en.wikipedia.org/wiki/K-means_clustering.
  The controller() function contains the logic of the algorithm as well as part of the visualization process, though we will cover that in the next step.
  The two stages of k-means are traditionally called Assignment and Update, so I named my functions Assignment() and Update() to keep it simple.
  
  Assignment: 
    For k clusters, there will be k new dataframes generated. 
    Each of the dataframes will contain a record of the euclidean distance from each point in the data to that particular centroid.
    After all distance dataframes are generated, a new dataframe is created which labels each point of the data with the cluster number for the closest centroid.
    
  Update:
    Now that every point is assigned to a cluster, the centroid of the cluster is repositioned at the center of all the points that are currently in its cluster.
    Important Note: If a cluster contains to points at this stage due to bad rng, it will cease to exist and not appear in any subsequent iterations.
    
  These steps are repeated by the controller() until it detects that no points have changed centroids from one iteration to the next, indicating accurate clusters.
  
3. Visualization

  Within the controller() function, generate_graph() is repeatedly executed.
  Every time is is called, it creates a jpeg of the current data and which points are in each cluster, then saves the jpeg to the specified folder.
  Odd behavior can be observed on the first image generated, but it resolves itself in subsequent stages, so I have deemed it sufficiently pleasing to look at.
  After controller() finishes execution and returns to the main(), make_gif() is called to generate our final result. 
  make_gif() employs the imageio library to compile each of the jpegs into a gif, which is then added to the same folder the images are in.
  
