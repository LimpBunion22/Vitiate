classdef RED
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        inputNum
        layerNum
        layers = cell(0);
    end
    
    methods
        function obj = RED(inputNum,neuronsPerLayer,layerNum)
            %UNTITLED3 Construct an instance of this class
            %   Detailed explanation goes here
            obj.inputNum = inputNum;
            obj.layerNum = layerNum;
            layers = cell(1,layerNum);
            for m=1:layerNum
                for n=1:neuronsPerLayer(m)
                    if m==1
                        layers{m}(n)=neurona(inputNum,1);
                    else
                        layers{m}(n)=neurona(neuronsPerLayer(m-1),1);
                    end
                end
            end
        end
        
        function outputs = forward(obj,inputs)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            for m=1:obj.layerNum
                npc = length(obj.layers{m});
                outputs = zeros(npc,1);
                for n=1:npc
                    outputs(n) = obj.layers{m}(n).algoritmo(inputs);
                end
                inputs = outputs;
            end
                    
        end
        
        function e = forward2(obj,inputs)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            e = cell(1,obj.layerNum+1);
            for m=1:obj.layerNum
                npc = length(obj.layers{m});
                outputs = zeros(npc,1);
                for n=1:npc
                    outputs(n) = obj.layers{m}(n).algoritmo(inputs);
                end
                e{m}=inputs;
                inputs = outputs;
            end
            e{obj.layerNum+1}=inputs;        
        end
        
        function output = gradiente(obj,inputs,Salidas)
            outputNum = length(obj.layers{obj.layerNum});
            output = cell(1,obj.layerNum);
            e = obj.fordward2(inputs);
            for m=1:obj.layerNum
                if m==1
                    output{m} = zeros(length(obj.layers{m}),obj.inputNum);
                else
                    output{m} = zeros(length(obj.layers{m}),length(obj.layers{m-1}));
                end
            end
            
            for s = 1:outputNum
                E = -2*(Salidas(s)-e{outputNum+1}(s))*obj.layers{outputNum}.Alfa(e{outputNum+1}(s));
                output{obj.layerNum}(s,1) = E;
                
                for n = 1:length(obj.layers{outputNum-1})
                    output{m}(s,n+1) = E*e{outputNum}(n);
                    
                    E2 = obj.layers{outputNum}.coefs(n)*obj.layers{outputNum-1}.Alfa(e{outputNum}(n));                    
                    output{obj.layerNum}(n,1) = E*E2 + output{obj.layerNum}(n,1);
                    
                    for c=1:obj.layers{outputNum-1}(1).inputNum
                        output{obj.layerNum}(n,c+1) = E*E2*e{outputNum-1}(c) + output{obj.layerNum}(n,c+1);
                    end
                end
                
                
            end
            
            
        end
    end
end

