function y = bilinearInterp(u,v,image)
f = floor(u) -u;
e = floor(v) -v;
y = (1-f)*(1-e)*image(int16(u),int16(v)) + f*(1-e)*image(int16(u)+1,int16(v)) +e*(1-f)*image(int16(u),int16(v)+1) +e*f*image(int16(u)+1,int16(v)+1);
end