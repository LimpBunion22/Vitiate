%PRIMER CASO DE TEST

%Datos

nData = 5000;
data = zeros(3,nData);
pltData = cell(4,1);
for i=1:nData
    switch randi(4)
        case 1
            data(1,i) = 2*rand();
            data(2,i) = 2*rand();
            data(3,i) = 1;
        case 2
            data(1,i) = -2*rand();
            data(2,i) = 2*rand();
            data(3,i) = 2;
        case 3
            data(1,i) = -2*rand();
            data(2,i) = -2*rand();
            data(3,i) = 3;
        case 4
            data(1,i) = 2*rand();
            data(2,i) = -2*rand();
            data(3,i) = 4;
        otherwise
    end
end

%%
mColor = [1 0 0; 0 1 0; 0 0 1; 0.75 0.25 01];
%'DisplayName',strcat("")
p = subplot(1,1,1);
hold on
for i=1:nData
    plot(data(1,i),data(2,i),'*','LineWidth',1.8,'color',mColor(data(3,i),:))
end
grid on
grid minor
%legend
set(p,'Fontsize',20)
hold off