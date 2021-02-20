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


myRED.layers{1}(1).beta = [-6.20239];
myRED.layers{1}(1).coefs = [-7.34497 -6.24512 7.97546 -8.85864];

myRED.layers{1}(2).beta = [-1.00403];
myRED.layers{1}(2).coefs = [8.87207 -7.56836 -8.32886 5.75012];

myRED.layers{1}(3).beta = [-6.54114];
myRED.layers{1}(3).coefs = [-5.53101 7.10571 7.96753 1.75476];
%-

myRED.layers{2}(1).beta = [3.21838];
myRED.layers{2}(1).coefs = [4.51904 -8.96484 7.7594];

myRED.layers{2}(2).beta = [-3.81836];
myRED.layers{2}(2).coefs = [2.89612 -6.48193 -0.770874];

myRED.layers{2}(3).beta = [9.74609];
myRED.layers{2}(3).coefs = [4.10095 6.74561 1.40198];
%-

myRED.layers{3}(1).beta = [0.513306];
myRED.layers{3}(1).coefs = [-8.61328 -8.20984 8.57117];

myRED.layers{3}(2).beta = [3.07495];
myRED.layers{3}(2).coefs = [-1.43372 4.44824 -8.36426];
%-

myRED.layers{4}(1).beta = [-0.141602];
myRED.layers{4}(1).coefs = [2.97485 8.85925];

myRED.layers{4}(2).beta = [-0.244141];
myRED.layers{4}(2).coefs = [8.67065 5.34363];
%-

myRED.layers{5}(1).beta = [-9.62646];
myRED.layers{5}(1).coefs = [4.22852 3.55713];

myRED.layers{5}(2).beta = [9.71863];
myRED.layers{5}(2).coefs = [8.31421 2.72156];



%%
myRED.forward([-2;1;10;-1])
%%
gradiente = myRED.gradiente([-2;1;10;-1],[4;-1])