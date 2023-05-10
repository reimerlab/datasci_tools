from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial import Delaunay
#import matplotlib.pyplot as plt
#import numpy as np
#import scipy

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

"""
Example of how to use polyfit----

****NOTE: makes ure the data doesn't have to have any vertical or horizontal shifts *****

"""
#from scipy.optimize import curve_fit
def model_fit(
    y,
    x=None,
    n_downsample=10,
    method = "poly4",
    ):
    

    # x = np.linspace(0, 3, 50)
    # y = np.exp(x)
    if n_downsample > 1:
        y = trace[::n_downsample]

    if x is None:
        x = np.arange(len(y)) + 1
    else:
        x = tracetimes[::n_downsample]


    """
    Plot your data
    """
    plt.plot(x, y, 'ro',label="Original Data")

    """
    brutal force to avoid errors
    """    
    x = np.array(x, dtype=float) #transform your data in a numpy array of floats 
    y = np.array(y, dtype=float) #so the curve_fit can work

    """
    create a function to fit with your data. a, b, c and d are the coefficients
    that curve_fit will calculate for you. 
    In this part you need to guess and/or use mathematical knowledge to find
    a function that resembles your data
    """
    
    def func_poly4(x, a, b, c, d,e):
        return a*x**4 + b*x**3 +c*x**2 + d*x + e
        return b*x**2 +c*x + d
    def func_poly6(x, a, b, c, d,e,f,g):
        return a*x**6 + b*x**5 +c*x**4 + d*x**3 +e*x**2 + f*x + g
    def func_exp(x, a, b, c):
        return a*np.exp(-b*x) + c
    
    func = eval(f"func_{method}")
    print(f"func = {func}")
    """
    make the curve_fit
    """
    popt, pcov = curve_fit(func, x, y,
                           #p0=[1,-1],
                           #p0 = [1,1]
                          )

    """
    The result is:
    popt[0] = a , popt[1] = b, popt[2] = c and popt[3] = d of the function,
    so f(x) = popt[0]*x**3 + popt[1]*x**2 + popt[2]*x + popt[3].
    """
    #print(f"{popt[0]}x^{popt[1]} + {popt[2]}")# % (popt[0], popt[1], popt[2], popt[3]))
    print(popt)

    """
    Use sympy to generate the latex sintax of the function
    """
    # xs = sym.Symbol('\lambda')    
    # tex = sym.latex(func(xs,*popt)).replace('$', '')
    # plt.title(r'$f(\lambda)= %s$' %(tex),fontsize=16)

    """
    Print the coefficients and plot the funcion.
    """

    plt.plot(x, func(x, *popt), label="Fitted Curve") #same as line above \/
    #plt.plot(x, popt[0]*x**3 + popt[1]*x**2 + popt[2]*x + popt[3], label="Fitted Curve") 

    plt.legend(loc='upper left')
    plt.show()
    
    
    return func(x, *popt)



