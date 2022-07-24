function [Zxx,Zyy,Z]=Integrability(Zx, Zy, height, width)
maxpq = 1;
lam = 1;
u = 3;
ZZx = zeros(height * 2, width * 2);
ZZx(1: height, 1: width) = Zx(:, :);
ZZx(height + 1: 2 * height, 1: width) = Zx(height: -1: 1, :);
ZZx(:, width + 1: width * 2) = ZZx(:, width: -1: 1);
ZZy = zeros(height * 2, width * 2);
ZZy(1: height, 1: width) = Zy(:, :);
ZZy(height + 1: 2 * height, 1 : width) = Zy(height: -1: 1, :);
ZZy(:, width + 1: width * 2) = ZZy(:, width: -1: 1);

Zx = ZZx;
Zy = ZZy;

height = height * 2;
width = width * 2;

for k = 2: width - 1,
    for l = 2: height - 1,
        if abs(Zx(l, k)) > maxpq | abs(Zy(l, k)) > maxpq
            Zx(l, k) = (Zx(l - 1, k) + Zx(l + 1, k) + Zx(l, k + 1) + Zx(l, k - 1)) / 4;
            Zy(l, k) = (Zy(l - 1, k) + Zy(l + 1, k) + Zy(l, k + 1) + Zy(l, k - 1)) / 4;
        end
    end
end

Cx = fft2(Zx, height, width);
Cy = fft2(Zy, height, width);

C = zeros(height,width); 
Cxx = C;
Cyy = C; 
% Cxx = zeros(height, width); Cyy = zeros(height, width);
for k = 1: width,
    for l = 1: height,
        wx = 2 * pi * (k - 1) / width;
        wy = 2 * pi * (l - 1) / height;
        if sin(wx) == 0 & sin(wy) == 0
            C(l, k)=0;
        else 
            cons = (1 + lam) * (sin(wx) ^ 2 + sin(wy) ^ 2) + u * (sin(wx) ^ 2 + sin(wy) ^ 2) ^ 2; 
            C(l, k) = (Cx(l, k) * (complex(0, -1) * sin(wx)) + Cy(l, k) * (complex(0, -1) * sin(wy))) / cons;
        end
        Cxx(l, k) = complex(0, 1) * sin(wx) * C(l, k);
        Cyy(l, k) = complex(0, 1) * sin(wy) * C(l, k);
    end
end
 height = height / 2;
 width = width / 2;
Z = real(ifft2(C));
Z = Z(1: height, 1: width);

Zxx = real(ifft2(Cxx));
Zyy = real(ifft2(Cyy));
Zxx = Zxx(1: height, 1: width);
Zyy = Zyy(1: height, 1: width);
maxpq = 1;
for k = 2: width - 1,
    for l = 2: height - 1,
        if abs(Zxx(l, k)) > maxpq | abs(Zyy(l, k)) > maxpq
            Zxx(l, k) = (Zxx(l - 1, k) + Zxx(l + 1, k) + Zxx(l, k + 1) + Zxx(l, k - 1)) / 4;
            Zyy(l, k) = (Zyy(l - 1, k) + Zyy(l + 1, k) + Zyy(l, k + 1) + Zyy(l, k - 1)) / 4;
            Z(l, k) = (Z(l - 1, k) + Z(l + 1, k) + Z(l, k + 1) + Z(l,k - 1)) / 4;
        end
    end
end


