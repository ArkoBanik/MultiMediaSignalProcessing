function Out = mouth_warp_ec(delta_w, delta_h1,delta_h2, fScale,image,file)

[im_height, im_width] = size(image);
fileID = fopen(file,'r');
formatSpec = '%d';
neutral_w = 1;
neutral_h1 = 1;
neutral_h2 = 2;
w = neutral_w +  delta_w;
h1 = neutral_h1 + delta_h1;
h2 = neutral_h2 + delta_h2;
numVerticies = str2num(fgetl(fileID));
sizeA = [2 numVerticies];
Mesh.vertex = fscanf(fileID,formatSpec,sizeA)';
fgetl(fileID);
fgetl(fileID);
numTriangles = str2num(fgetl(fileID));
sizeB = [3 numTriangles];
Mesh.triangles = fscanf(fileID,formatSpec,sizeB)';

[Newmesh.vertex(:,1), Newmesh.vertex(:,2)]= interpVert(Mesh.vertex(:,1),Mesh.vertex(:,2),neutral_w, neutral_h1, neutral_h2, w, h1, h2, fScale);

% make X_k's 
for i =1:numTriangles
   ogTriangle(i,1:2,:) = Mesh.vertex(Mesh.triangles(i,1:3),1:2)';
   ogTriangle(i,3,:) = [1 1 1];
   newTriangles(i,1:2,:) = Newmesh.vertex(Mesh.triangles(i,1:3),1:2)';
   newTriangles(i,3,:) = [1 1 1];
end

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

Out= mat2gray(out_image);

end