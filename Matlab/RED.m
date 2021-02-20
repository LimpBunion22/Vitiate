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
            obj.layers = cell(1,layerNum);
            for m=1:layerNum
                for n=1:neuronsPerLayer(m)
                    if m==1
                        obj.layers{m}(n)=neurona(inputNum,1);
                    else
                        obj.layers{m}(n)=neurona(neuronsPerLayer(m-1),1);
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
            outputRow = zeros(outputNum,length(obj.layers{obj.layerNum-1}));
            output = cell(1,obj.layerNum);
            e = obj.forward2(inputs);
            for m=1:obj.layerNum
                if m==1
                    output{m} = zeros(length(obj.layers{m}),obj.inputNum+1);
                else
                    output{m} = zeros(length(obj.layers{m}),length(obj.layers{m-1})+1);
                end
            end
            
            for s = 1:outputNum
                E = -2*(Salidas(s)-e{obj.layerNum+1}(s))*obj.layers{obj.layerNum}(s).Alfa(e{obj.layerNum+1}(s));
                output{obj.layerNum}(s,1) = E;
                
                for n = 1:length(obj.layers{obj.layerNum-1})
                    output{m}(s,n+1) = E*e{obj.layerNum}(n);
                    
                    E2 = obj.layers{obj.layerNum}(s).coefs(n)*obj.layers{obj.layerNum-1}(n).Alfa(e{obj.layerNum}(n));                    
                    output{obj.layerNum-1}(n,1) = E*E2 + output{obj.layerNum-1}(n,1);                    
                    output{obj.layerNum-1}(n,2:(obj.layers{obj.layerNum-1}(1).inputNum+1)) = transpose(E*E2*e{obj.layerNum-1}) + output{obj.layerNum-1}(n,2:(obj.layers{obj.layerNum-1}(1).inputNum+1));
                end
                
                outputRow(s,:) = obj.layers{obj.layerNum}(s).coefs*E;
            end
            
            if obj.layerNum>2
                for n=1:length(obj.layers{obj.layerNum-2})
                    alfa = obj.layers{obj.layerNum-2}(n).Alfa(e{obj.layerNum-1}(n));
                    col = zeros(length(obj.layers{obj.layerNum-1}),1);
                    for n2=1:length(obj.layers{obj.layerNum-1})
                        col(n2) = obj.layers{obj.layerNum-1}(n2).coefs(n)*alfa;
                    end
                    [A,~] = obj.buildMatrix(obj.layerNum-1,e);
                    for s = 1:outputNum
                        val = outputRow(s,:)*A*col;
                        output{obj.layerNum-2}(n,1) = val + output{obj.layerNum-2}(n,1);
                        output{obj.layerNum-2}(n,2:(obj.layers{obj.layerNum-2}(1).inputNum+1)) = transpose(val*e{obj.layerNum-2}) + output{obj.layerNum-2}(n,2:(obj.layers{obj.layerNum-2}(1).inputNum+1));
                    end
                end
            end
            
            if obj.layerNum>3
                result = eye(length(obj.layers{obj.layerNum-1}));
                for m = obj.layerNum-3:-1:1
                    [A,C] = obj.buildMatrix(m+2,e);
                    result = result * A * C;
                    for n = 1:length(obj.layers{m})
                        alfa = obj.layers{m}(n).Alfa(e{m+1}(n));
                        col = zeros(length(obj.layers{m+1}),1);
                        for n2=1:length(obj.layers{m+1})
                            col(n2) = obj.layers{m+1}(n2).coefs(n) * alfa * obj.layers{m+1}(n2).Alfa(e{m+2}(n2));
                        end
                        result2 = result*col;
                        for s = 1:outputNum
                            val = outputRow(s,:)*result2;
                            output{m}(n,1) = val + output{m}(n,1);
                            output{m}(n,2:(obj.layers{m}(1).inputNum+1)) = transpose(val*e{m}) + output{m}(n,2:(obj.layers{m}(1).inputNum+1));
                        end
                    end
                end
            end
            
            
        end
        
        function [A,C] = buildMatrix(obj,layer,e)
            A = eye(length(obj.layers{layer}));
            C = zeros(length(obj.layers{layer}),obj.layers{layer}(1).inputNum);
            for i = 1: length(obj.layers{layer})
                A(i,:)= A(i,:)*obj.layers{layer}(i).Alfa(e{layer}(i));
                C(i,:)= obj.layers{layer}(i).coefs;
            end
        end
    end
end

