clc;clear;close all;

dir1='~/dsb18/model/';
dir_out = '~/dsb18/output/';
dir_img=dir([dir1 '*.png']);
dir_mask=dir([dir1 '*.npy']);
colors = [250 200 80];

se = ones(3,3);

num_subfigures = 3;

for i= 1:size(dir_mask)
    id = dir_mask(i).name(1:end-9);
    I = imread([dir_img(i).folder '/' dir_img(i).name]);
    M = readNPY([dir_mask(i).folder '/' dir_mask(i).name]);
    I = I(:,1:size(I,2)/num_subfigures,:);
    tmp1=I(:,:,1);
    tmp2=I(:,:,2);
    tmp3=I(:,:,3);
    if sum(tmp1(:)-tmp2(:))==0 & sum(tmp2(:)-tmp2(:))==0
        gray_image = 1;
    else
        gray_image = 0;
    end
    
    V=unique(M(:));
    V(1)=[];
    Mnew = zeros(size(M));
    
    mean_size = sum(M(:)>0)/length(V);
    
    if ismember(i,[6 8 15 25])
        counts = imhist(rgb2gray(I), 16);
        % Compute a global threshold using the histogram counts.
        T = otsuthresh(counts);
        % Binarize image using computed threshold.
        BW = imbinarize(rgb2gray(I),T);
        if sum((BW(:)==1).*(M(:)>0)) > sum((BW(:)==0).*(M(:)>0))
            fg_label = 1;
        else
            fg_label = 0;
        end;
        bg_label = 1-fg_label;
        fg_ratio = zeros(length(V),1);
    end;
    
    for n=1:length(V)
        tmp=zeros(size(M));
        tmp(M==V(n))=1;
        
        if gray_image
            tmp_d=imclose(tmp,se);
        else
            tmp_d=imdilate(tmp,se);
        end
        
        tmp_f=imfill(tmp_d);
        tmp_f(1,:)=tmp_f(2,:);
        tmp_f(:,1)=tmp_f(:,2);
        tmp_f(end,:)=tmp_f(end-1,:);
        tmp_f(:,end)=tmp_f(:,end-1);
        
        tmp_f= bwareafilt(tmp_f>0,1);
        
        if sum(tmp_f(:)>0)/mean_size <= 10 %3.5
            Mnew(tmp_f>0)=V(n);
        end
        if ismember(i,[6 8 15 25])
            fg_ratio(n) =sum((M(:)==V(n)).*(BW(:)==fg_label))/sum(M(:)==V(n));
        end
    end
    
    if ismember(i,[6 8 15 25])
        figure;imagesc(I);pause;
        fg_ratio_low = find(fg_ratio<0.7);
        Mnew_edit = Mnew;
        for k = 1:length(fg_ratio_low)
            Mnew_edit(Mnew==V(fg_ratio_low(k)))=0;
        end
        Mnew_edit_uni = unique(Mnew_edit);
        Mnew_edit_uni(1)=[];
        Mnew = zeros(size(Mnew));
        for k = 1:length(Mnew_edit_uni)
            Mnew(Mnew_edit==Mnew_edit_uni(k))=k;
        end
    end;
    
    V=unique(Mnew(:));
    V(1)=[];
    R = zeros(length(V),1);
    for n=1:length(V)
        R(n)=sum(Mnew(:)==V(n))/mean_size;
    end
    disp([min(R) max(R) std(R)])
    
    figure(1);
    subplot(1,3,1);imshow(I);
    Im = I;
    for z=1:3
        tmp = Im(:,:,z);
        tmp(M>0)=colors(z);
        Im(:,:,z)=tmp;
    end
    subplot(1,3,2);imshow(Im);
    
    Im = I;
    for z=1:3
        tmp = Im(:,:,z);
        tmp(Mnew>0)=colors(z);
        Im(:,:,z)=tmp;
    end
    subplot(1,3,3);imshow(Im);
    
    I = imread([dir_img(i).folder '/' dir_img(i).name]);
    I(:,size(I,2)/num_subfigures*2+1:end,:) = Im;
%     imwrite(I, [dir_img(i).folder '/' dir_img(i).name]);
    
    writeNPY(Mnew, [dir_out  dir_mask(i).name]);
end