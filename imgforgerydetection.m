close all; clear all;
im1='10.jpg'
[OutputMap, Feature_Vector, coeffArray] = cal(im1);
imagesc(OutputMap);

function [F1Map,CFADetected, F1] = CFATamperDetection_F1(im)    
    StdThresh=5;
    Depth=3;
    
    im=double(im(1:round(floor(end/(2^Depth))*(2^Depth)),1:round(floor(end/(2^Depth))*(2^Depth)),:));
    
    % list of possible CFA arrangements
    SmallCFAList={[2 1;3 2] [2 3;1 2] [3 2;2 1] [1 2;2 3]};
    
    CFAList=SmallCFAList;
    
    %block size
    W1=16;
    
    if size(im,1)<W1 || size(im,2)<W1
        F1Map=zeros([size(im,1), size(im,2)]);
        CFADetected=[0 0 0 0];
        return
    end
    
    MeanError=inf(length(CFAList),1);
    for TestArray=1:length(CFAList)
        
        BinFilter=[];
        ProcIm=[];
        CFA=CFAList{TestArray};
        R=CFA==1;
        G=CFA==2;
        B=CFA==3;
        BinFilter(:,:,1)=repmat(R,size(im,1)/2,size(im,2)/2);
        BinFilter(:,:,2)=repmat(G,size(im,1)/2,size(im,2)/2);
        BinFilter(:,:,3)=repmat(B,size(im,1)/2,size(im,2)/2);
        CFAIm=double(im).*BinFilter;
        BilinIm=bilinInterp(CFAIm,BinFilter,CFA);
        
        
        ProcIm(:,:,1:3)=im;
        ProcIm(:,:,4:6)=double(BilinIm);
        
        ProcIm=double(ProcIm);
        BlockResult=blockproc(ProcIm,[W1 W1],@eval_block);
        
        Stds=BlockResult(:,:,4:6);
        BlockDiffs=BlockResult(:,:,1:3);
        NonSmooth=Stds>StdThresh;
        
        MeanError(TestArray)=mean(mean(mean(BlockDiffs(NonSmooth))));
        BlockDiffs=BlockDiffs./repmat(sum(BlockDiffs,3),[1 1 3]);
        
        Diffs(TestArray,:)=reshape(BlockDiffs(:,:,2),1,numel(BlockDiffs(:,:,2)));
        F1Maps{TestArray}=BlockDiffs(:,:,2);
    end
    
    Diffs(isnan(Diffs))=0;
    
    [~,val]=min(MeanError);
    U=sum(abs(Diffs-0.25),1);
    F1=median(U);
    CFADetected=CFAList{val}==2;
    F1Map=F1Maps{val};
    
end

function [ Out ] = eval_block( block_struc )
    im=block_struc.data;
    Out(:,:,1)=mean2((double(block_struc.data(:,:,1))-double(block_struc.data(:,:,4))).^2);
    Out(:,:,2)=mean2((double(block_struc.data(:,:,2))-double(block_struc.data(:,:,5))).^2);
    Out(:,:,3)=mean2((double(block_struc.data(:,:,3))-double(block_struc.data(:,:,6))).^2);
    
    Out(:,:,4)=std(reshape(im(:,:,1),1,numel(im(:,:,1))));
    Out(:,:,5)=std(reshape(im(:,:,2),1,numel(im(:,:,2))));
    Out(:,:,6)=std(reshape(im(:,:,3),1,numel(im(:,:,3))));
end



function [F1Map,CFADetected, F1] = cal( imPath )
    im=CleanUpImage(imPath);
    [F1Map,CFADetected, F1] = CFATamperDetection_F1(im);
end


function [ Out_Im ] = bilinInterp( CFAIm,BinFilter,CFA )

MaskMin=1/4*[1 2 1;2 4 2;1 2 1];
MaskMaj=1/4*[0 1 0;1 4 1;0 1 0];

if ~isempty(find(diff(CFA)==0)) || ~isempty(find(diff(CFA')==0))
    MaskMaj=MaskMaj.*2;
end

Mask=repmat(MaskMin,[1,1,3]);
[a,Maj]=max(sum(sum(BinFilter)));
Mask(:,:,Maj)=MaskMaj;

Out_Im=zeros(size(CFAIm));

for ii=1:3
    Mixed_im=zeros([size(CFAIm,1),size(CFAIm,2)]);
    Orig_Layer=CFAIm(:,:,ii);
    Interp_Layer=imfilter(Orig_Layer,Mask(:,:,ii));
    Mixed_im(BinFilter(:,:,ii)==0)=Interp_Layer(BinFilter(:,:,ii)==0);
    Mixed_im(BinFilter(:,:,ii)==1)=Orig_Layer(BinFilter(:,:,ii)==1);
    Out_Im(:,:,ii)=Mixed_im;
end

Out_Im=uint8(Out_Im);