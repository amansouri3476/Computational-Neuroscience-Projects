function output = TrialExtractionmodified(s)
n = 205;
indeces = IndExtractionmodified(s);

trialtestnum = length(indeces);
channelnum = 11;
% test
temp = repmat(indeces,[n,1]) + repmat((0:n-1)', [1,trialtestnum]);
temp = temp(:)';
output = reshape(s.test(:,temp),[channelnum n trialtestnum]);


end