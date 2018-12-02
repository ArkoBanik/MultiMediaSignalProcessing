%test code for image warping
image = imread('mouth.jpg');
[im_height, im_width] = size(image);
fileID = fopen('mesh.txt','r');
formatSpec = '%d';
neutral_w = 1;
neutral_h1 = 1;
neutral_h2 = 1;
fScale = 2;
w = neutral_w +  av_train.visual(1,88);
h1 = neutral_h1 + av_train.visual(2,88);
h2 = neutral_h2 + av_train.visual(3,88);
numVerticies = str2num(fgetl(fileID));
sizeA = [2 numVerticies];
Mesh.vertex = fscanf(fileID,formatSpec,sizeA)';
fgetl(fileID);
fgetl(fileID);
numTriangles = str2num(fgetl(fileID));
sizeB = [3 numTriangles];
Mesh.triangles = fscanf(fileID,formatSpec,sizeB)';

[Newmesh.vertex(:,1), Newmesh.vertex(:,2)]= interpVert(mesh.vertex(:,1),mesh.vertex(:,2),neutral_w, neutral_h1, neutral_h2, w, h1, h2, fScale);
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
num_pixels_in_triangle = 0;
out_image = zeros(im_height, im_width);
for x =1:im_height
   for y =1:im_width 
        pixel = [x y 1]';
        lambda = temp*pixel;
        out_pixel = 0;
        if(all(lambda >= 0 & lambda <=1))
            num_pixels_in_triangle = num_pixels_in_triangle +1;
            u_vect = permute(ogTriangle(k,:,:),[2 3 1])*lambda;
            u = u_vect(1);
            v = u_vect(2);
            out_pixel = bilinearInterp(u,v,image)
        end
        out_image(x,y) = out_pixel;
   end
end

figure()
imshow(out_image)