# Alignment and Stability of Embeddings

 _**embas**sy_: **emb**edding **a**lignment and **s**tability

This Python package provides tools 
* to measure alignment and stability errors of given embeddings,
* to align given embeddings.

The package will soon be updated with new functionality for extracting rotation angles of a rotation matrix, creating rotation matrices with a desired magnitude, adding random walk-based noise with a desired magnitude, and other operations.

## Dependecies

1. [numpy](https://numpy.org/)
2. [scipy](https://www.scipy.org/)

Tested for numpy==1.20.1 and scipy==1.6.1 but should work with most versions.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install _embassy_.

```bash
pip install embassy
```

## Example Usage

```python
from embassy import align_and_measure
import numpy as np

X = np.array([[1.8, 2.0], [2.3,  2.5], [ 1.8, 4.2],  [4.1, 3.1]])
Y = np.array([[0.0, 1.2], [0.3,  1.6], [-0.4, 3.5 ], [1.6, 2.5]])

output = align_and_measure(X, Y)
             
print("\n Translation Error :", output['translation_error'], 
      "\n Rotation Error    :", output['rotation_error'],    
      "\n Scale Error       :", output['scale_error'],       
      "\n Stability Error   :", output['stability_error'],
      "\n",
      "\n X_aligned:\n",        output['emb1'],
      "\n",
      "\n Y_aligned:\n",        output['emb2'])
```

See _Examples/Demonstration.ipynb_ for visual and real-world examples.

## Citation

If you find this software useful in your work, please cite:

Furkan Gursoy, Mounir Haddad, and Cecile Bothorel. (2021). ["Alignment and stability of embeddings: measurement and inference improvement."](https://arxiv.org/abs/2101.07251) (2020).



## Contributing

Please feel free to open an issue for bug reports, change requests, or other contributions.


## License

[MIT](https://choosealicense.com/licenses/mit/)

Packaged with: [Flit](https://buildmedia.readthedocs.org/media/pdf/flit/latest/flit.pdf)
