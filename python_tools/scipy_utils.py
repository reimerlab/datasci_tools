from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import scipy

# ------ 3/10: Delauny Triangulation -------#
"""
Notes on the Delauny triangulation:
1) just makes a surface that encloses all points in a triangulation
such that no point in P is inside the circumcircle of any triangle in DT(P).

The simplicies just group the vertex indices into the groups that make up the triangle
Ex: 
array([[2, 3, 0],
        [3, 1, 0]]
        
        
find_simplex --> Find the simplices containing the given points (will return the simplices index)

"""

def example_delaunay_triangulation(points=None,
                                   plot_triangulation=True,
                                   plot_shaded_triangulation=True,
                                  verbose=False,):
    if points is None:
        points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1],[5,2],[-1,3]])
    tri = Delaunay(points)
    
    if verbose:
        print(f"Vertices grouped into triangles")
        
    if plot_triangulation:
        plt.triplot(points[:,0], points[:,1], tri.simplices)
        plt.plot(points[:,0], points[:,1], 'o')
        plt.show()
        
    if plot_shaded_triangulation:
        N = 6
        random_color_faces = np.random.choice(np.linspace(0,1,N),size=len(tri.simplices),replace=True)
        fig,ax = plt.subplots()
        ax.tripcolor(points[:,0], points[:,1], tri.simplices, 
                     facecolors=random_color_faces
                     #color=["red"]*len(points[:,0]) + 
                     , edgecolors='black',cmap="Greens")
        plt.show()
        

def linear_regression(x,y,verbose = False):
    slope, intercept, r_value, p_value, std_err =scipy.stats.linregress(x,y)
    if verbose:
        print(f"slope = {slope}")
        print(f"intercept = {intercept}")
        print(f"r_value = {r_value}")
        print(f"p_value = {p_value}")
    return [slope,intercept]