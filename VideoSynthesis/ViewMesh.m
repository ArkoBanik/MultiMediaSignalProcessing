fileID = fopen('mesh.txt','r');
formatSpec = '%d';
neutral_w = 1;
neutral_h1 = 1;
neutral_h2 = 1;
fScale = 5;
w = neutral_w +  av_train.visual(1,88);
h1 = neutral_h1 + av_train.visual(2,88);
h2 = neutral_h2 + av_train.visual(3,88);
numVerticies = str2num(fgetl(fileID));
sizeA = [2 numVerticies];
mesh.vertex = fscanf(fileID,formatSpec,sizeA)';
mesh.tri = delaunay(mesh.vertex(:,1),mesh.vertex(:,2));

[newmesh.vertex(:,1), newmesh.vertex(:,2)]= interpVert(mesh.vertex(:,1),mesh.vertex(:,2),neutral_w, neutral_h1, neutral_h2, w, h1, h2, fScale);
newmesh.tri = delaunay(newmesh.vertex(:,1),newmesh.vertex(:,2));

figure()
imshow(imread('mouth.jpg'));

hold on;

trimesh(mesh.tri,mesh.vertex(:,1),mesh.vertex(:,2));
scatter(mesh.vertex(:,1),mesh.vertex(:,2));

%figure()

%imshow(imread('mouth.jpg'));
%hold on;
%trimesh(newmesh.tri,newmesh.vertex(:,1),newmesh.vertex(:,2));
%scatter(newmesh.vertex(:,1),newmesh.vertex(:,2));

%%
fileID = fopen('adiMesh.txt','r');
formatSpec = '%d';
neutral_w = 1;
neutral_h1 = 1;
neutral_h2 = 1;
fScale = 5;
w = neutral_w +  av_train.visual(1,88);
h1 = neutral_h1 + av_train.visual(2,88);
h2 = neutral_h2 + av_train.visual(3,88);
numVerticies = str2num(fgetl(fileID));
sizeA = [2 numVerticies];
mesh.vertex = fscanf(fileID,formatSpec,sizeA)';
mesh.vertex(:,2) = mesh.vertex(:,2) +2;
mesh.tri = delaunay(mesh.vertex(:,1),mesh.vertex(:,2));

figure()
y = 61;
x = 100;
admouth = rgb2gray(imread('adi_mouth.jpg'));
%admouth(y,x) = 255;
imshow(admouth);
% imshow(rgb2gray(imread('adi_mouth.jpg')));

hold on;

trimesh(mesh.tri,mesh.vertex(:,1),mesh.vertex(:,2));
scatter(mesh.vertex(:,1),mesh.vertex(:,2));