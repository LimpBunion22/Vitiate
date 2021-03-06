%%
%Script para las baterías de prueba

myNrn = neurona(3,1);
%%
myNrn.algoritmo([1;-20;3])
%%
inputNum = 4;
layerNum = 5;
neuronsPerLayer = [3,3 ,2,2,2]; 
myRED = RED(inputNum,neuronsPerLayer,layerNum)
%%
myRED.layers{1}(1).beta = [-7.1405];
myRED.layers{1}(1).coefs = [5.24841 5.71472 5.62622 9.33533 ];

myRED.layers{1}(2).beta = [-4.24194];
myRED.layers{1}(2).coefs = [-9.80347 -3.31116 5.00488 -4.9707 ];

myRED.layers{1}(3).beta = [-5.01099];
myRED.layers{1}(3).coefs = [-9.83826 -5.95276 0.497437 -2.12891 ];

myRED.layers{2}(1).beta = [-7.55981];
myRED.layers{2}(1).coefs = [8.06641 -5.02502 7.52869 ];

myRED.layers{2}(2).beta = [-3.88794];
myRED.layers{2}(2).coefs = [3.3728 -2.68555 -5.94788 ];

myRED.layers{2}(3).beta = [0.955811];
myRED.layers{2}(3).coefs = [0.48584 -3.60168 9.47632 ];

myRED.layers{3}(1).beta = [4.82056];
myRED.layers{3}(1).coefs = [-6.01013 -5.93018 1.48376 ];

myRED.layers{3}(2).beta = [-4.98169];
myRED.layers{3}(2).coefs = [0.865479 -7.3645 -6.59973 ];

myRED.layers{4}(1).beta = [0.843506];
myRED.layers{4}(1).coefs = [5.42786 -7.43652 ];

myRED.layers{4}(2).beta = [-6.40381];
myRED.layers{4}(2).coefs = [-2.146 8.98132 ];

myRED.layers{5}(1).beta = [-3.99963];
myRED.layers{5}(1).coefs = [7.43896 4.53247 ];

myRED.layers{5}(2).beta = [2.33704];
myRED.layers{5}(2).coefs = [1.73584 -5.10376 ];






%%
myRED.forward([-2;1;10;-1])
%%
gradiente = myRED.gradiente([-2;1;10;-1],[4;-1])