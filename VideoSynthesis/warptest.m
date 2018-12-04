%test code for image warping
image = imread('mouth.jpg');
[im_height, im_width] = size(image);
fileID = fopen('mesh.txt','r');
formatSpec = '%d';
neutral_w = 1;
neutral_h1 = 1;
neutral_h2 = 2;
fScale = 1.5;
w = neutral_w +  av_train.visual(1,1597);
h1 = neutral_h1 + av_train.visual(2,1597);
h2 = neutral_h2 + av_train.visual(3,1597);
numVerticies = str2num(fgetl(fileID));
sizeA = [2 numVerticies];
Mesh.vertex = fscanf(fileID,formatSpec,sizeA)';
fgetl(fileID);
fgetl(fileID);
numTriangles = str2num(fgetl(fileID));
sizeB = [3 numTriangles];
Mesh.triangles = fscanf(fileID,formatSpec,sizeB)';

[Newmesh.vertex(:,1), Newmesh.vertex(:,2)]= interpVert(Mesh.vertex(:,1),Mesh.vertex(:,2),neutral_w, neutral_h1, neutral_h2, w, h1, h2, fScale);
%ogTriangle = zeros(numTriangles,3,3);

% make X_k's 
for i =1:numTriangles
   ogTriangle(i,1:2,:) = Mesh.vertex(Mesh.triangles(i,1:3),1:2)';
   ogTriangle(i,3,:) = [1 1 1];
   newTriangles(i,1:2,:) = Newmesh.vertex(Mesh.triangles(i,1:3),1:2)';
   newTriangles(i,3,:) = [1 1 1];
end

% example to turn kth triangle into 3x3 shape and check each pixel against
% it
% k is kth triangle for this example Im using k = 1
k = 1;
temp = inv(permute(newTriangles(k,:,:),[2 3 1]));
out_image = zeros(im_height, im_width);
for y =1:im_height
   for x =1:im_width
        pixel = [x y 1]';
        lambda = temp*pixel;
        out_pixel = 0;
        if(all(lambda >= 0 & lambda <=1))
            u_vect = permute(ogTriangle(k,:,:),[2 3 1])*lambda;
            u = u_vect(1);
            v = u_vect(2);
            out_pixel = bilinearInterp(u,v,image);
        end
        out_image(x,y) = out_pixel;
   end
end

figure()
imshow(out_image)
%%
close all
out_image = zeros(im_height, im_width);
for y =1:im_height
   for x =1:im_width
       for k=1:numTriangles
            X_inv =  inv(permute(newTriangles(k,:,:),[2 3 1]));
            pixel = [x y 1]';
            lambda = X_inv*pixel;
            out_pixel = uint8(out_image(y,x));
            if(all(lambda >= 0 & lambda <=1))
                u_vect = permute(ogTriangle(k,:,:),[2 3 1])*lambda;
                u = u_vect(1);
                v = u_vect(2);
                out_pixel = uint8(bilinearInterp(u,v,image));
            end
            out_image(y,x) = out_pixel;
       end
   end
end

out= mat2gray(out_image);
figure()
imshow(out)
figure()
imshow(image)
%% Test the function implementation
load('ECE417_MP4_AV_Data.mat');
image = imread('mouth.jpg');
test_point = 1800;
delta_w  = av_train.visual(1,test_point);
delta_h1 = av_train.visual(2,test_point);
delta_h2 = av_train.visual(3,test_point);
fScale = 1.5;

Out = mouth_warp(delta_w, delta_h1,delta_h2, fScale,image);

out_image_name = 'test.jpg';
destinationFolder = './out_images';
fullDestinationFileName = fullfile(destinationFolder, out_image_name);

imwrite(Out, fullDestinationFileName);
imwrite(Out,out_image_name)

figure()
imshow(Out)
%% Test Extra Credit with Adi's face
load('ECE417_MP4_AV_Data.mat');
image = rgb2gray(imread('adi_mouth.jpg'));
file = 'adiMesh.txt';
test_point = 1800;
delta_w  = av_train.visual(1,test_point);
delta_h1 = av_train.visual(2,test_point);
delta_h2 = av_train.visual(3,test_point);
fScale = 1.5;

Out = mouth_warp_ec(delta_w, delta_h1,delta_h2, fScale,image,file);

% out_image_name = 'test.jpg';
% destinationFolder = './out_images';
% fullDestinationFileName = fullfile(destinationFolder, out_image_name);
% 
% imwrite(Out, fullDestinationFileName);
% imwrite(Out,out_image_name)

figure()
imshow(Out)