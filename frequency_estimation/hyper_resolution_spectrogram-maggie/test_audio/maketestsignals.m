%%% makes test signals (middle a, with modulations)

% by Dr. Marcelo Magnasco at The Rockefeller University
% included as part of an example in the hyperresolution-spectrogram routine
% maintained by Maggie Zhang <jiayizha@andrew.cmu.edu>

SR=96000; 

t = (1:4*SR) / SR; 
x = asin( sin(2*pi*440*t))/pi + 1e-4*randn(1,4*SR); 
audiowrite('middle a triangle.wav',x,SR); 

%%
d = 1; 
m = sin(2*pi*55*t); 
x = asin( sin(2*pi*440*t + d*m   ))/pi +   1e-4*randn(1,4*SR); 

audiowrite('middle a fm 55 1.wav',x,SR); 

%%

d = 2; 
m = sin(2*pi*22*t); 
x = asin( sin(2*pi*440*t + d*m   ))/pi +   1e-4*randn(1,4*SR); 

audiowrite('middle a fm 22 2.wav',x,SR); 


%%

d = 5; 
m = sin(2*pi*110*t); 
x = asin( sin(2*pi*440*t + d*m   ))/pi +   1e-4*randn(1,4*SR); 

audiowrite('middle a fm 11 5.wav',x,SR); 