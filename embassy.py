""" Python package for aligning embeddings and measuring alignment and stability errors."""
__version__ = '0.1'


import numpy as np
import numpy.linalg as LA
from scipy.linalg import orthogonal_procrustes, null_space

def align_and_measure(A, B):
    """Aligning embeddings and measuring alignment and stability errors.
	
    Alignment and stability errors are calculated, and embedding matrices are aligned based on the methods and measures provided in the following paper. Please cite it if you find this software useful in your work.
            
    Furkan Gursoy, Mounir Haddad, and Cecile Bothorel. "Alignment and stability of embeddings: measurement and inference improvement." (2021). https://arxiv.org/abs/2101.07251
    
    Parameters
    ----------
    A : ndarray
        A two-dimensional numpy ndarray of shape (n_objects, dimensionality_of_embedding).
        Rows and columns should have the same order as B.
        
    B : ndarray
        A two-dimensional numpy ndarray of shape (n_objects, dimensionality_of_embedding).
        Rows and columns should have the same order as A.
    
    Returns
    -------
    Returns a dictionary with the following keys.
    
    translation_error : float
        Translation error between the input embeddings A and B.
        
    rotation_error : float
        Rotation error between the input embeddings A and B.
        
    scale_error : float
        Scale error between the input embeddings A and B.
        
    stability_error : float
        Stability error between the input embeddings A and B.
        
    emb1 : ndarray
        Aligned (translated and rotated) version of A.
              
    emb2 : ndarray
        Aligned (translated and rotated) version of B.
    
    Notes
    -------
    
    For usage examples, please visit the project's main page at https://github.com/furkangursoy/embassy
	"""
    try:
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise Exception("A and B must be two dimensional.")
    except:
        print("A and B must be two dimensional.")
        
        
    if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]:
        raise Exception("The shapes of A and B must match.")
    
    mtx1, mtx2 = A.copy().astype(float), B.copy().astype(float)
    
    
    #calculate the center point of the embeddings
    center1 = np.mean(mtx1, 0)
    center2 = np.mean(mtx2, 0)

    #center them at the origin
    mtx1 -= center1
    mtx2 -= center2

    
    #calculate radii
    radius1 = np.mean(LA.norm(mtx1, axis = 1))
    radius2 = np.mean(LA.norm(mtx2, axis = 1))
    
    #calculate translation error
    translation_error = 1 - 1/( 1 + (LA.norm(center1-center2) / (radius1 + radius2)))
    
    #calculate rotation error
    d = mtx1.shape[1]
    tmp_mtx1 = mtx1.copy()
    tmp_mtx2 = mtx2.copy()
    if LA.matrix_rank( np.concatenate([mtx1, mtx2]), tol=10**-10) != d:
        new_vecs = null_space(np.concatenate([A, B]), rcond=10**-10).T
        tmp_mtx1 = np.concatenate([mtx1, new_vecs])
        tmp_mtx2 = np.concatenate([mtx2, new_vecs])
    R, _= orthogonal_procrustes(tmp_mtx2, tmp_mtx1)
    rotation_error =  np.sqrt(np.sum((R-np.identity(d))**2)/(d**2))/(2/np.sqrt(d))
    
    #rotate
    mtx2 = mtx2@R
    
    #store embeddings before scaling operations
    emb1 = mtx1.copy()
    emb2 = mtx2.copy()
    
    #calculate scale error
    scale_error = np.abs(radius1 - radius2) / np.average([radius1, radius2]) / 2

    #calculate stability error
    mtx1 /= radius1
    mtx2 /= radius2
    stability_error = np.mean(LA.norm(mtx1 - mtx2, axis = 1)/(LA.norm(mtx1, axis = 1) + LA.norm(mtx2, axis = 1)))
    
    
    return {'translation_error' : translation_error,
              'rotation_error'    : rotation_error,
              'scale_error'       : scale_error,
              'stability_error'   : stability_error,
              'emb1'              : emb1,
              'emb2'              : emb2,          
         }