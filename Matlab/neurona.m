classdef neurona
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        algoIndex
        inputNum
        beta = randn();
        coefs
    end
    
    methods
        function obj = neurona(inputNum,algoIndex)
            obj.inputNum = inputNum;
            obj.algoIndex = algoIndex;
            obj.coefs = randn(1,inputNum);
        end
        
        function sumatorio = algoritmo(obj,inputs)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            sumatorio = obj.beta+obj.coefs*inputs;
            switch obj.algoIndex
                case 1
                    if sumatorio<0
                        sumatorio = sumatorio/256;
                    end
                otherwise
                    sumatorio = 0;
                    disp("Error: Unknown Algorithm")
            end            
        end
        
        function A = Alfa(obj,res)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
            switch obj.algoIndex
                case 1
                    if res<0
                        A = 1/256;
                    else
                        A = 1;
                    end
                otherwise
                    A = 0;
                    disp("Error: Unknown Algorithm")
            end            
        end
    end
end

