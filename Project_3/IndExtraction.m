function output = IndExtraction(s)

target_train = find(diff(s.train(11,:)) == 1) + 1;
ntarget_train = find(diff(s.train(10,:).*(1 - s.train(11,:))) > 0) + 1;
target_test = find(diff(s.test(11,:)) == 1) + 1;
ntarget_test = find(diff(s.test(10,:).*(1 - s.test(11,:))) > 0) + 1;
output = struct('target_train',target_train,'ntarget_train',ntarget_train,'target_test',target_test,'ntarget_test',ntarget_test);

end