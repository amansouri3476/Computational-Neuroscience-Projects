function [hdr,record,output] = FeatureExtraction(edfFileAddress,hypFileAddress)
    
    [hdr,record] = edfread(edfFileAddress);
    hypnogram = AnnotExtract(hypFileAddress);
    
    % Clearing Data from outliers
    
    clearIndex = [(find(record(5,:)==2)) , (find(record(5,:)==1))];
    temp = ( repmat((clearIndex - 1)*10 , [10, 1]) + repmat((1:10)', [1, length(clearIndex)]) );
    clearIndeces = temp(:);
    record = record(:, clearIndeces);

      
      record(:,abs(record(1,:))>mean(record(1,:))+5*std(record(1,:))) = [];
      record(:,abs(record(2,:))>mean(record(2,:))+5*std(record(2,:))) = [];
      record(:,abs(record(3,:))>mean(record(3,:))+5*std(record(3,:))) = [];
      record(:,abs(record(4,:))>mean(record(4,:))+5*std(record(4,:))) = [];
    % Equalizing the length to a multiple of ten
    
    record(:,floor(length(record(1,:))/10)*10+1:end) = [];
    
    % BPF impulse responses
    
    Fs = 100; % According to data set descriptions

    % lower and upper frequencies adapted from Wikipedia

    
    deltaBPF = BPF(1001,0.01,4,Fs);
    thetaBPF = BPF(1001,4,7,Fs);
    alphaBPF = BPF(1001,8,15,Fs);
    betaBPF = BPF(1001,16,31,Fs);
    close all % to close figures opening after above codes run
    record(:,1+1000*floor(length(record(1,:))/1000):end)=[];
    t = nan(length(record(1,:))/1000, 1);
    state = nan(length(record(1,:))/1000, 1);

    
    for i = 1 : length(record(1,:))/1000
        t(i,1) = (i-1)*10;
        state(i,1) = hypnogram(2,find(10*i>hypnogram(1,:), 1, 'last' ));
        
    end
        
    reshapeRecord_1 = reshape(record(1,:),[1000,floor(length(record(1,:))/1000)]);
       
    FpzDeltaFilt = FilterDFT(reshapeRecord_1,deltaBPF);
    FpzThetaFilt = FilterDFT(reshapeRecord_1,thetaBPF);
    FpzAlphaFilt = FilterDFT(reshapeRecord_1,alphaBPF);
    FpzBetaFilt = FilterDFT(reshapeRecord_1,betaBPF); 
        
    FpzDelta = bandpower(FpzDeltaFilt);
    FpzTheta = bandpower(FpzThetaFilt);
    FpzAlpha = bandpower(FpzAlphaFilt);
    FpzBeta = bandpower(FpzBetaFilt);
    
    for i = 1 : size(reshapeRecord_1, 2)
        FpzDeltaMC(1,i) = max(abs(myfft(FpzDeltaFilt(:,i), Fs, 'dontplot')));
        FpzThetaMC(1,i) = max(abs(myfft(FpzThetaFilt(:,i), Fs, 'dontplot')));
        FpzAlphaMC(1,i) = max(abs(myfft(FpzAlphaFilt(:,i), Fs, 'dontplot')));
        FpzBetaMC(1,i) = max(abs(myfft(FpzBetaFilt(:,i), Fs, 'dontplot')));
    end
    
    reshapeRecord_2 = reshape(record(2,:),[1000,floor(length(record(1,:))/1000)]);
        
    OzDeltaFilt = FilterDFT(reshapeRecord_2,deltaBPF);
    OzThetaFilt = FilterDFT(reshapeRecord_2,thetaBPF);
    OzAlphaFilt = FilterDFT(reshapeRecord_2,alphaBPF);
    OzBetaFilt = FilterDFT(reshapeRecord_2,betaBPF);
        
    OzDelta = bandpower(OzDeltaFilt);
    OzTheta = bandpower(OzThetaFilt);
    OzAlpha = bandpower(OzAlphaFilt);
    OzBeta = bandpower(OzBetaFilt);
    
    for i = 1 : size(reshapeRecord_2, 2)
        OzDeltaMC(1,i) = max(abs(myfft(OzDeltaFilt(:,i), Fs, 'dontplot')));
        OzThetaMC(1,i) = max(abs(myfft(OzThetaFilt(:,i), Fs, 'dontplot')));
        OzAlphaMC(1,i) = max(abs(myfft(OzAlphaFilt(:,i), Fs, 'dontplot')));
        OzBetaMC(1,i) = max(abs(myfft(OzBetaFilt(:,i), Fs, 'dontplot')));
    end
        
    reshapeRecord_3 = reshape(record(3,:),[1000,floor(length(record(1,:))/1000)]);
        
    EOGPower = bandpower(reshapeRecord_3);
    
    reshapeRecord_4 = reshape(record(4,:),[1000,floor(length(record(1,:))/1000)]);
       
    EMGPower = bandpower(reshapeRecord_4);
        
    X = [FpzDelta',FpzTheta',FpzAlpha',FpzBeta',OzDelta',OzTheta',OzAlpha',OzBeta',EOGPower',EMGPower'];
    
    X_added = ([FpzDeltaMC',FpzThetaMC',FpzAlphaMC',FpzBetaMC',OzDeltaMC',OzThetaMC',OzAlphaMC',OzBetaMC'].^2)./X(:,1:8);
    
    
    output = struct('times',t,'states',state,'X',X,'X_added',X_added);
    
    
end
