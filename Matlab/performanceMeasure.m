%PERFORMANCE

inputNum = 4;
layerNum = 5;
neuronsPerLayer = [5,5,5,5,5]; 
myRED = RED(inputNum,neuronsPerLayer,layerNum);

%%

f = @() myRED.forward([-10;10;-10;10]);
g = @()myRED.gradiente([-10;10;-10;10],[-10;10;-10;10;10]);
tf = timeit(f)*10^6
tg = timeit(g)*10^6

%%
iN = 2;
lN = 5;
nPL = 2;
for i=2:50
    nPL=i;
    neuronsPerLayer = nPL*ones(1,lN);
    myRED = RED(iN,neuronsPerLayer,lN);
    
    in = randn(iN,1);
    out = randn(nPL,1);
    
    f = @() myRED.forward(in);
    g = @()myRED.gradiente(in,out);
    
    tf(i,1)=i;
    tf(i,2)=timeit(f);
    
    tg(i,1)=i;
    tg(i,2)=timeit(g);
end

p = subplot(1,1,1);
hold on 
mColor = [1 0 0; 0 1 0; 0 0 1; 0.75 0.15 0.05];

plot(tf(:,1),tf(:,2),'LineWidth',1.8,'DisplayName',"Forward")
plot(tg(:,1),tg(:,2),'LineWidth',1.8,'DisplayName',"Backward")

%plot(datetime(1:length(SP),'ConvertFrom','posixtime','Format','dd HH:mm:ss'),SP-transpose(DeadBand(:,1)),'DisplayName',strcat("MInf : ",string(MInf)),'LineStyle','- -','LineWidth',1.8,'color',[1 0.5 0])

grid on
grid minor
legend
set(p,'Fontsize',20)
hold off