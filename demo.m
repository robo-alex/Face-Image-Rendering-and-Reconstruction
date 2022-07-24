clear all; close all; clc;
dirname='dataset_offline';
dirs=dir([dirname,'/P*']);
for i=1:size(dirs,1)
    pth=[dirname, '/',dirs(i).name];
    [z,imgs]=rendering(pth);
    save([pth,'/z'],'z');
    for j=1:size(imgs,3)
        img=imgs(:,:,j);
        imwrite(img,[pth,'/',num2str(j),'.bmp']);
    end
end
