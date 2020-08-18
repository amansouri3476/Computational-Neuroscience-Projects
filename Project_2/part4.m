disp('===================== part 4 =====================')
%% =============== Q1 ===============
disp(' =============== Q1 ===============')

% extracting NREM states
NREMIndeces = (subject{subNumber}.states >= 1) & (subject{subNumber}.states <= 4); % states 1 to 4 correspond to NREM
XNREM = subject{subNumber}.X( NREMIndeces, :);
sNREM = subject{subNumber}.states(NREMIndeces); % states of NREM


% fitting linear model to raw data
linMod = fitlm(XNREM, sNREM);
disp(' ')
disp(' ------- linMod for raw data -------')
disp(linMod)

% fitting linear model to normalized data
nNREM = size(XNREM, 1);
XNREMNorm = (XNREM - repmat(mean(XNREM),[nNREM, 1]))./repmat(std(XNREM), [nNREM, 1]);
linModNorm = fitlm(XNREMNorm, sNREM);
disp(' ')
disp(' ------- linMod for normalized data -------')
disp(linModNorm)

% investigating predictors effect sizes
disp(' ')
disp(' ------- R2 effect size (same for raw or normalized)  -------')
effect_size_in_linMdl(XNREM, sNREM, 'print');

disp(' ')
disp(' ------- coeff effect size for normalized data  -------')
coeffNORM = table2array(linModNorm.Coefficients(2:size(linModNorm.Coefficients, 1), 'Estimate'));
[sorted_eff_size, sorted_indeces] = sort(abs(coeffNORM) , 'descend');
disp('var   eff-size')
for i = 1 : size(XNREM, 2)
    disp(sprintf('%s    %.3f',linModNorm.VariableNames{sorted_indeces(i)}, coeffNORM(sorted_indeces(i)))) 
end

%% =============== Q4 ===============
disp(' =============== Q4 ===============')
figure
hold on
histogram(linMod.Fitted(sNREM == 1), 'Normalization', 'probability', 'BinMethod', 'scott')
histogram(linMod.Fitted(sNREM == 2), 'Normalization', 'probability', 'BinMethod', 'scott')
histogram(linMod.Fitted(sNREM == 3), 'Normalization', 'probability', 'BinMethod', 'scott')
histogram(linMod.Fitted(sNREM == 4), 'Normalization', 'probability', 'BinMethod', 'scott')
legend('1', '2', '3', '4')
axis([0 4 0 0.4])
title(sprintf('fitted states(depth) distribution - subject %d', subNumber))

%% =============== Q5 ===============
disp(' =============== Q5 ===============')

% extracting REM data
REMIndeces = (subject{subNumber}.states == 6); % state 6 corresponds to REM
XREM = subject{subNumber}.X(REMIndeces, :);

% extracting WAKE data
WIndeces = (subject{subNumber}.states == 0); % state 0 corresponds to WAKE
XW = subject{subNumber}.X(WIndeces, :);

% prediction
sREM_predicted = predict(linMod, XREM);
sW_predicted = predict(linMod, XW);

% distribution
figure
hold on
histogram(sREM_predicted, 'Normalization', 'probability', 'BinMethod', 'scott')
histogram(sW_predicted, 'Normalization', 'probability', 'BinMethod', 'scott')
legend('REM', 'WAKE')
title(sprintf('predicted sleep depth for WAKE and REM stages - subject %d', subNumber))

[HREMW, pREMW] = ttest2(sREM_predicted, sW_predicted);
[HREM, pREM] = ttest2(sREM_predicted, linMod.Fitted);
[HW, pW] = ttest2(sW_predicted, linMod.Fitted);
