function output = TrialExtraction(s)
n = 205;
indeces = IndExtraction(s);
trialtargettrainnum = length(indeces.target_train);
trialntargettrainnum = length(indeces.ntarget_train);
trialtargettestnum = length(indeces.target_test);
trialntargettestnum = length(indeces.ntarget_test);
channelnum = 11;
% target_train
temp = repmat(indeces.target_train,[n,1]) + repmat((0:n-1)', [1,trialtargettrainnum]);
temp = temp(:)';
target_train = reshape(s.train(:,temp),[channelnum n trialtargettrainnum]);

% ntarget_train
temp = repmat(indeces.ntarget_train,[n,1]) + repmat((0:n-1)', [1,trialntargettrainnum]);
temp = temp(:)';
ntarget_train = reshape(s.train(:,temp),[channelnum n trialntargettrainnum]);

% target_test
temp = repmat(indeces.target_test,[n,1]) + repmat((0:n-1)', [1,trialtargettestnum]);
temp = temp(:)';
target_test = reshape(s.test(:,temp),[channelnum n trialtargettestnum]);

% ntarget_test
temp = repmat(indeces.ntarget_test,[n,1]) + repmat((0:n-1)', [1,trialntargettestnum]);
temp = temp(:)';
ntarget_test = reshape(s.test(:,temp),[channelnum n trialntargettestnum]);

output = struct('target_train',target_train,'ntarget_train',ntarget_train,'target_test',target_test,'ntarget_test',ntarget_test);

end