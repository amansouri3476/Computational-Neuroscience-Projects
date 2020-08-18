function [Y,f]=myfft(x,Fs,dontplot)
    %[Y,F] = myfft(x,Fs,dontplot)
    %claculates x(t) fourier series that assumed to be clipped and
    %sampled by Fs frequency
    %Y: x fourier series
    %F: fequencies
    %Fs: sampling rate
    X=fft(x);
    N=length(x);
    X=[X(2:N);X(1)];
    fr=(1:floor((N+1)/2))/N*Fs;
    fl=fr-(N-floor(N/2))/N*Fs;
    f=[fl';fr'];
    Yr=1/N*X(1:floor((N+1)/2)); %right part of fourier series
    Yl=1/N*X(floor((N+2)/2):N); %left part of fourier series
    Y=[Yl;Yr];
    if (nargin == 2)
        figure
        plot(f,abs(Y))
        xlabel('f(Hz)')
    end
end