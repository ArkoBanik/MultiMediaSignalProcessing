% Trains Neural Network and Generates frames for video
load('ECE417_MP4_AV_Data.mat');


%% Train Network
resultFile = 'trained_network.mat';
avTrainingData = av_train;
avValidateData = av_validate;
numN = 300;
mapping = ECE417_MP4_train ( avTrainingData, avValidateData, silenceModel, numN, resultFile );

%% Test Network
[results] = ECE417_MP4_test ( testAudio, silenceModel, mapping );



%% Generate Images
[f, num_frames] = size(results);
image = imread('mouth.jpg');
fScale = 1.5;
destinationFolder = './out_images';

for i=1:num_frames
    delta_w  = results(1,i); 
    delta_h1 = results(2,i);
    delta_h2 = results(3,i);
    Out = mouth_warp(delta_w, delta_h1,delta_h2, fScale,image);
    
    out_image_name = strcat('test_', sprintf('%04d',i),'.jpg');    
    fullDestinationFileName = fullfile(destinationFolder, out_image_name);
    imwrite(Out, fullDestinationFileName);
end

%%
a = 443;
delta_w  = results(1,a); 
delta_h1 = results(2,a);
delta_h2 = results(3,a);
Out = mouth_warp(delta_w, delta_h1,delta_h2, fScale,image);

%out_image_name = strcat('test_', sprintf('%04d',a),'.jpg');    
%fullDestinationFileName = fullfile(destinationFolder, out_image_name);
%imwrite(Out, fullDestinationFileName);

    
  
 
  
  
    