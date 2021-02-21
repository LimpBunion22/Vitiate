%Test 1

nDatos = 5000;
datos = zeros(3,nDatos);
for i = 1:nDatos
    switch randi(4)
        case 1
            datos(1,i) = 0.5 + 0.2^rand();
            datos(2,i) = 0.5 + 0.2^rand();
            datos(3,i) = 1;
        case 2
            datos(1,i) = -0.5  -0.2^rand();
            datos(2,i) = 0.5 + 0.2^rand();
            datos(3,i) = 2;
        case 3
            datos(1,i) = -0.5  -0.2^rand();
            datos(2,i) = -0.5  -0.2^rand();
            datos(3,i) = 3;
        case 4
            datos(1,i) = 0.5 + 0.2^rand();
            datos(2,i) = -0.5  -0.2^rand();
            datos(3,i) = 4;
        otherwise
    end
end


%%

p = subplot(1,1,1);
hold on 
mColor = [1 0 0; 0 1 0; 0 0 1; 0.75 0.15 0.05];
for i=1:nDatos
plot(datos(1,i),datos(2,i),'*','LineWidth',1.8,'color',mColor(datos(3,i),:))
end
%plot(datetime(1:length(SP),'ConvertFrom','posixtime','Format','dd HH:mm:ss'),SP-transpose(DeadBand(:,1)),'DisplayName',strcat("MInf : ",string(MInf)),'LineStyle','- -','LineWidth',1.8,'color',[1 0.5 0])

grid on
grid minor
%legend
set(p,'Fontsize',20)
hold off
%%
%CUIDADDO REINICIA ---------------------------------
inputNum = 2;
layerNum = 5;
neuronsPerLayer = [4,4,4,4,4]; 
myRED = RED(inputNum,neuronsPerLayer,layerNum);

iteraciones = 5;
%%





error = 0;
for i = 1:nDatos
    res = myRED.forward(datos(1:2,i));
    [~,pltRes(i)] =  max(res);
    if pltRes(i)~=datos(3,i)
        error = error+1;
    end
end
disp(error)

p = subplot(1,1,1);
hold on 
mColor = [1 0 0; 0 1 0; 0 0 1; 0.75 0.15 0.05];
for i=1:nDatos
plot(datos(1,i),datos(2,i),'*','LineWidth',1.8,'color',mColor(pltRes(i),:))
end
%plot(datetime(1:length(SP),'ConvertFrom','posixtime','Format','dd HH:mm:ss'),SP-transpose(DeadBand(:,1)),'DisplayName',strcat("MInf : ",string(MInf)),'LineStyle','- -','LineWidth',1.8,'color',[1 0.5 0])

grid on
grid minor
%legend
set(p,'Fontsize',20)
hold off
%%
grdAc = cell(layerNum,1);
for it = 1:iteraciones
    disp(strcat("Iteracion: ",string(it)))
   for i = 1:nDatos
       RS = zeros(4,1);
       RS(datos(3,i)) =1;
       
       gradiente = myRED.gradiente(datos(1:2,i),RS);   
       
       for m=1:layerNum
           if i==1
               grdAc{m}=gradiente{m};
           else
               grdAc{m}=grdAc{m}+gradiente{m};
           end
       end
   end
norma = 0;
 for m=1:layerNum
     for n=1:length(grdAc{m}(:,1))
         for c=1:length(grdAc{m}(1,:))
             norma = norma + grdAc{m}(n,c)^2;
         end
     end
 end
 norma = sqrt(norma);
 for m=1:layerNum
     for n=1:length(grdAc{m}(:,1))
         for c=1:length(grdAc{m}(1,:))
             if c==1
                 myRED.layers{m}(n).beta = myRED.layers{m}(n).beta -  0.02/iteraciones*grdAc{m}(n,c)/norma;
             else
                 myRED.layers{m}(n).coefs(c-1)= myRED.layers{m}(n).coefs(c-1) - 0.02/iteraciones*grdAc{m}(n,c)/norma;
             end
         end
     end
 end
end
%%



p = subplot(1,1,1);
hold on 
mColor = [1 0 0; 0 1 0; 0 0 1; 0.75 0.15 0.05];
for x = -2:0.1:2
    for y = -2:0.1:2
        res = myRED.forward([x;y]);
        [~,ac] =  max(res);
        
        plot(x,y,'*','LineWidth',1.8,'color',mColor(ac,:))
    end
end
%plot(datetime(1:length(SP),'ConvertFrom','posixtime','Format','dd HH:mm:ss'),SP-transpose(DeadBand(:,1)),'DisplayName',strcat("MInf : ",string(MInf)),'LineStyle','- -','LineWidth',1.8,'color',[1 0.5 0])

grid on
grid minor
%legend
set(p,'Fontsize',20)
hold off

%%
prevError = 5001;
error = 5000;
grdAc = cell(layerNum,1);
it = 0;
while prevError>=error    
    prevError = error;
    it = it+1;
    
   for i = 1:nDatos
       RS = zeros(4,1);
       RS(datos(3,i)) =1;
       
       gradiente = myRED.gradiente(datos(1:2,i),RS);   
       
       for m=1:layerNum
           if i==1
               grdAc{m}=gradiente{m};
           else
               grdAc{m}=grdAc{m}+gradiente{m};
           end
       end
   end
norma = 0;
 for m=1:layerNum
     for n=1:length(grdAc{m}(:,1))
         for c=1:length(grdAc{m}(1,:))
             norma = norma + grdAc{m}(n,c)^2;
         end
     end
 end
 norma = sqrt(norma);
 for m=1:layerNum
     for n=1:length(grdAc{m}(:,1))
         for c=1:length(grdAc{m}(1,:))
             
                 if c==1
                     myRED.layers{m}(n).beta = myRED.layers{m}(n).beta -  0.05*grdAc{m}(n,c)/norma;
                 else
                     myRED.layers{m}(n).coefs(c-1)= myRED.layers{m}(n).coefs(c-1) - 0.05*grdAc{m}(n,c)/norma;
                 end
             
         end
     end
 end
 error = 0;
for i = 1:nDatos
    res = myRED.forward(datos(1:2,i));
    [~,pltRes(i)] =  max(res);
    if pltRes(i)~=datos(3,i)
        error = error+1;
    end
end
disp(strcat("Iteracion: ",string(it)," Error: ",string(error)))
end