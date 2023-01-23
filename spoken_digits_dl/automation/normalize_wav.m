clear all
d1 = dir('/nethome/pinheirs/Documents/papi/easyspeech/mots8/mots8_update/dataset_justsounds/');
str1 = {d1.name};
nbfile=length(str1)-2;

for m=1:nbfile
nom_fichier=['/nethome/pinheirs/Documents/papi/easyspeech/mots8/mots8_update/dataset_justsounds/',str1{m+2}]
    [sig,fs] = audioread(nom_fichier);
    msig=max(abs(sig));
    slim = [-msig/0.9 msig/0.9];
    dx=diff(slim);
    sigsc = (sig-slim(1))/dx*2-1;
    audiowrite(['/nethome/pinheirs/Documents/papi/easyspeech/mots8/mots8_update/dataset_justsounds/',str1{m+2}], sigsc,fs);
end