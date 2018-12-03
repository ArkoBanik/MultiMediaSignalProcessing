function y = bilinearInterp(u,v,image)
[max_v, max_u] = size(image);
f = u -floor(u);
e = v -floor(v);
y = 0;
if floor(u) <= max_u & floor(v) <= max_v
    y = (1-f)*(1-e)*image(floor(v),floor(u)) ;
    if floor(u)+1 <= max_u
    y= y+ f*(1-e)*image(floor(v),floor(u)+1) ;
    end
    if floor(v)+1 <= max_v
    y = y +e*(1-f)*image(floor(v)+1,floor(u));
    end
    if floor(u)+1 <= max_u & floor(v)+1 <= max_v
    y = y + e*f*image(floor(v)+1,floor(u)+1);
    end
end

end