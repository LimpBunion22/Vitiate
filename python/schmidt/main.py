import numpy as np
from BaseElement import base, myfunc

#!NECESEARIO NUMPY Y MATPLOTLIB¡

# El código es una primera aproximación y no es muy legible, pero desde
# aquí se pude testear fácilmente

# Paso de integración para los productos escalares
step = 0.1

# Dominio de las funciones [-domaine,domaine]
domaine = 128

# Orden, para distribuir los bias se va dividiendo el dominio de dos en dos -> 2^order neuronas base
order = 5

# La funcion test a proyectar, sin() en este caso, pero se pude cambiar a placer
yfunc = np.zeros(int(2*domaine/step))
yfunc = np.sin(np.arange(-domaine,domaine,step)/10)

# Genera un objeto que contiene los coeficientes y valores de las funciones ortonormalizadas
myElement = myfunc(yfunc)
myBase = base(step, domaine, order)

# Grafica la base ortonormal
myBase.graph_base()

# Proyecta la función en la base
coefs = myBase.proyect_function(myElement)

# Grafica la función proyectada
myBase.graph_representation(yfunc, coefs)