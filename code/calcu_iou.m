% tic;
% thes = 0.4;
bb = aboxes;
c=1;
for thes = 0.5:0.5
    iou_re=[];
    iou_pre=[];
    iou_pr=[];
    tic;
    for iou = 0.05:0.05:1
        iou
        for i = 1:1980
            im = e6_iou{i};
            im2 = double(double(im)/max(max(double(im))));
            im2(im2>thes)=1;
            im2(im2<=thes)=0;
            myrecall1(i) = 0;
            myrecall2(i) = 0;
            myrecall3(i) = 0;
            for k = 1:200
                im3 = zeros(size(im2));
                try a = round(bb{i}(k,1:4));
                catch 
                    a = round(bb{i}(185,1:4));
                end
                im3(a(2):a(4),a(1):a(3))=1;
                t1 = (im2 & im3);
                t2 = (im2 | im3);
                tt= length(find(t1==1))/length(find(t2==1));
                if(tt>iou)
                    if k<51
                        myrecall1(i) = 1;
                    elseif k<101
                        myrecall2(i) = 1;
                    elseif k<201
                        myrecall3(i) = 1;
                        break;
                    end
                end
            end
        end
        %50
        imgrecall(c,1) = length( find(myrecall1==1))/1980;
        %50
        imgrecall(c,2) = length( find(myrecall2==1))/1980;
        %50
        imgrecall(c,3) = length( find(myrecall3==1))/1980;
        imgrecall
        c = c+1;
    end
    toc;
end


%    iou_re(i,k) = length(find(t1==1))/length(find(im2==1));
%    iou_pre(i,k) = length(find(t1==1))/length(find(im3==1));
%    final(c,1) = length( find(myrecall1==1))/1980
%    final(c,2) = length( find(myrecall2==1))/1980
%    final(c,3) = length( find(myrecall3==1))/1980
%    result(c,1)=mean(iou_re(:));
%    result(c,2)=mean(iou_pre(:));
%    result(c,3)=mean(iou_pr(:));