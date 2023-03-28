clear all; close all; clc;

for m=5:5

modelName = sprintf('UNet_2X_CCA_model_%d.mat',m);
load(modelName)
for i = 1:length(imdsTest.Files)
    file_name = cell2mat(imdsTest.Files(i)); 
    testImage = imread(file_name);
    imshow(testImage)
    if m == 1
       C = semanticseg(testImage,UNet_2X_CCA_model_1);
       elseif m==2
       C = semanticseg(testImage,UNet_2X_CCA_model_2);
       elseif m==3
       C = semanticseg(testImage,UNet_2X_CCA_model_3); 
       elseif m==4
       C = semanticseg(testImage,UNet_2X_CCA_model_4);
       elseif m==5
       C = semanticseg(testImage,UNet_2X_CCA_model_5); 
       elseif m==6
       C = semanticseg(testImage,UNet_2X_CCA_model_6); 
       elseif m==7
       C = semanticseg(testImage,UNet_2X_CCA_model_7); 
       elseif m==8
       C = semanticseg(testImage,UNet_2X_CCA_model_8); 
       elseif m==9
       C = semanticseg(testImage,UNet_2X_CCA_model_9);
       elseif m==10
       C = semanticseg(testImage,UNet_2X_CCA_model_10); 
    end

    label_overlay_img = labeloverlay(testImage,C,'IncludedLabels',"Plaque",...
        'Colormap','autumn','Transparency',0.8);
    h = imshow(label_overlay_img);
    newStr = extractAfter(file_name,"(1)-2X-Raw\");
    img_name = extractBefore(newStr,'.jpg');
    imwrite(h.CData, strcat(pwd,'\2-Overlay\', strcat(img_name,'-overlayed.png')));
     RGB = label2rgb(C,'gray');
    imshow(RGB)
    newStr = extractAfter(file_name,"(1)-2X-Raw\");
    img_name = extractBefore(newStr,'.jpg');
    h1 = imshow(~(mat2gray(rgb2gray(RGB))));
    imwrite(h1.CData, strcat(pwd,'\1-MaskOut\', strcat(img_name,'-maskOut.png')));
    B = labeloverlay(testImage,C,'IncludedLabels',"Plaque",...
        'Colormap','autumn','Transparency',0.8);
    GT_File = cell2mat(pxdsTest.Files(i))
    GT = imread(GT_File);
%     imshow(GT)
    precision1(i) = sum(sum(h1.CData&GT))/sum(h1.CData(:)); %% Sensitivity
    recall1(i) = sum(sum(h1.CData&GT))/sum(GT(:))*100;     %% PPV

    [Accuracy(i), Sensitivity(i),Specificity(i), Fmeasure(i), Precision(i), MCC(i), Dice(i), Jaccard(i)] = EvaluateImageSegmentationScores(h1.CData, GT);

   mean_precision = mean(precision1)*100;
   mean_recall = mean(recall1)*100;
   
   
end
all_metrics = [Accuracy', Sensitivity',Specificity', Fmeasure', Precision', MCC', Dice', Jaccard']
filename = sprintf('UNet_2X_CCA_model_%d_metrics',m)
xlswrite(filename,all_metrics);
clear i Accuracy' Sensitivity' Specificity' Fmeasure' Precision' MCC' Dice' Jaccard'


%     if m == 1
%       pixdsResults = semanticseg(imdsTest,UNet_2X_CCA_model_1);
%       UNet_2X_CCA_model_1_metrics = evaluateSemanticSegmentation(pixdsResults,pxdsTest);
%       save UNet_2X_CCA_model_1_metrics
% 
%        elseif m==2
%        pixdsResults = semanticseg(imdsTest,UNet_2X_CCA_model_2);
%        UNet_2X_CCA_model_2_metrics = evaluateSemanticSegmentation(pixdsResults,pxdsTest);
%        save UNet_2X_CCA_model_2_metrics
% 
%        elseif m==3
%             pixdsResults = semanticseg(imdsTest,UNet_2X_CCA_model_3); 
%             UNet_2X_CCA_model_3_metrics = evaluateSemanticSegmentation(pixdsResults,pxdsTest);
%             save UNet_2X_CCA_model_3_metrics
% 
%        elseif m==4
%            pixdsResults = semanticseg(imdsTest,UNet_2X_CCA_model_4);
%             UNet_2X_CCA_model_4_metrics = evaluateSemanticSegmentation(pixdsResults,pxdsTest);
%             save UNet_2X_CCA_model_4_metrics
%        
%        elseif m==5
%            pixdsResults = semanticseg(imdsTest,UNet_2X_CCA_model_5); 
%             UNet_2X_CCA_model_5_metrics = evaluateSemanticSegmentation(pixdsResults,pxdsTest);
%             save UNet_2X_CCA_model_5_metrics
% 
%        elseif m==6
%            
%            pixdsResults = semanticseg(imdsTest,UNet_2X_CCA_model_6); 
%             UNet_2X_CCA_model_6_metrics = evaluateSemanticSegmentation(pixdsResults,pxdsTest);
%             save UNet_2X_CCA_model_6_metrics
% 
%        elseif m==7
%            
%            pixdsResults = semanticseg(imdsTest,UNet_2X_CCA_model_7); 
%             UNet_2X_CCA_model_7_metrics = evaluateSemanticSegmentation(pixdsResults,pxdsTest);
%             save UNet_2X_CCA_model_7_metrics
% 
%        elseif m==8
%            
%           pixdsResults = semanticseg(imdsTest,UNet_2X_CCA_model_8); 
%             UNet_2X_CCA_model_8_metrics = evaluateSemanticSegmentation(pixdsResults,pxdsTest);
%             save UNet_2X_CCA_model_8_metrics
% 
%        elseif m==9
%          pixdsResults = semanticseg(imdsTest,UNet_2X_CCA_model_9);
%             UNet_2X_CCA_model_9_metrics = evaluateSemanticSegmentation(pixdsResults,pxdsTest);
%             save UNet_2X_CCA_model_9_metrics
% 
%        elseif m==10
%             
%             pixdsResults = semanticseg(imdsTest,UNet_2X_CCA_model_10); 
%             UNet_2X_CCA_model_10_metrics = evaluateSemanticSegmentation(pixdsResults,pxdsTest);
%             save UNet_2X_CCA_model_10_metrics
% 
%     end





% for i = 1:length(imdsTest.Files)
%     file_name = cell2mat(imdsTest.Files(i)) 
%     testImage = imread(file_name);
%     imshow(testImage)
%     C = semanticseg(testImage,UNet_2X_CCA_model_1);
%     RGB = label2rgb(C,'gray');
%     imshow(RGB)
%     newStr = extractAfter(file_name,"(15)Resized-Raw-Images128\");
%     img_name = extractBefore(newStr,'.jpg');
%     h1 = imshow(~(mat2gray(rgb2gray(RGB))));
% %     imwrite(h1.CData, strcat(end_results, strcat(img_name,'-maskOut.png')));
%     B = labeloverlay(testImage,C,'IncludedLabels',"Plaque",...
%         'Colormap','autumn','Transparency',0.75);
%     GT_File = cell2mat(pxdsTest.Files(i))
%     GT = imread(GT_File);
% %     imshow(GT)
%     precision1(i) = sum(sum(h1.CData&GT))/sum(h1.CData(:)); %% Sensitivity
%     recall1(i) = sum(sum(h1.CData&GT))/sum(GT(:))*100;     %% PPV
% %     h2 = imshow(B);
% %     imwrite(h2.CData, strcat(end_results, strcat(img_name,'-overlayed.png')));
%     [Accuracy(i), Sensitivity(i),Specificity(i), Fmeasure(i), Precision(i), MCC(i), Dice(i), Ja2X_CCArd(i)] = EvaluateImageSegmentationScores(h1.CData, GT);
% 
%    mean_precision = mean(precision1)*100;
%    mean_recall = mean(recall1)*100;
% end

%% Testing and evaluation of matrics over Image datastore (many images)
%%

end
   