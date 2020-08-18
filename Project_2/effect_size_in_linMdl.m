function eff_size = effect_size_in_linMdl(X, y, print)
% investigates effect size of each predictor on how well a linear model
% fits to data.
% inputs:
%   X: predictors, rows correspond to observations
%   y: response to be fitted, should have same number of rows as X
% output:
%   eff_size(i): how much R squared is increased by adding ith predictor 
    m = size(X, 2); % num of features
    
    linMod_tot = fitlm(X, y);
    R2_tot = linMod_tot.Rsquared.Ordinary;
    
    eff_size = nan(1, m);
    for i = 1 : m
        indeces = [(1 : i - 1) , (i + 1 : m)];
        linMod = fitlm(X(:,indeces), y);
        R2 = linMod.Rsquared.Ordinary;
        
        eff_size(i) = R2_tot - R2;
    end
    
    if (nargin > 2) && strcmp(print, 'print')
        [sorted_eff_size, sorted_indeces] = sort(eff_size , 'descend');
        disp('var   eff-size')
        for i = 1 : m
            disp(sprintf('%s    %.3f',linMod_tot.VariableNames{sorted_indeces(i)}, sorted_eff_size(i))) 
        end
    end
    
end
        
