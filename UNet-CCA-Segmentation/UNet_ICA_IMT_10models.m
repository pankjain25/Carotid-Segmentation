clear all; close all; clc;


input_image_path = "G:\(11)-AI-in-MEDICINE-Journal\(0)-Database\(0)-ICA\(1)-224x224\(1)-Raw-224x224";
mask_image_path = "G:\(11)-AI-in-MEDICINE-Journal\(0)-Database\(0)-ICA\(1)-224x224\(0)-Mask-224x224";
end_results = pwd;
% cd (input_image_path)
% image_files=dir('*.jpg');
% cd(pwd)
% cd(mask_image_path)
% mask_files = dir('*.png');

%% Prepare training, testing and validation dataset (images) with ground-truth values (B_box labels)
%% and shuffle them. Take 80% training 10% testing and 10% validation images with ground truth.
%%

encoderDepth = 4;
% lgraph = UNetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth)
classNames = ["Plaque","Background"];
labelIDs   = [255 0];
imageSize = [224 224 3];
numClasses = 2;


imageDir = input_image_path ;
labelDir = mask_image_path ;

imds = imageDatastore(imageDir);
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
fold =5;

c = cvpartition(imds.Files,'KFold',fold)

for Mod = 1:5 %fold
    
idx1 = training(c, Mod);
idx2 = test(c, Mod);

[imdstrain, imdsTest, pxdstrain, pxdsTest] = partitionData2(imds,pxds,labelIDs,idx1,idx2);
[imdsTrain,imdsVal, pxdsTrain,pxdsVal] = partitionData(imdstrain,pxdstrain,labelIDs);
  
Test_files=  imdsTest.Files(1:end);


combined_Datastore_Train = pixelLabelImageDatastore(imdsTrain,pxdsTrain);
combined_Datastore_Val = pixelLabelImageDatastore(imdsVal,pxdsVal);
combined_Datastore_Test = pixelLabelImageDatastore(imdsTest,pxdsTest);


lgraph = unetlgraph();
plot(lgraph)

options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',combined_Datastore_Val,...
    'MaxEpochs',100,'MiniBatchSize',10,...
    'VerboseFrequency',5,...
    'Plots','training-progress');

    if Mod == 1
            xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
            UNet_2X_ICA_model_1 = trainNetwork(combined_Datastore_Train,lgraph,options)
            save UNet_2X_ICA_model_1
        elseif Mod == 2
            xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
            UNet_2X_ICA_model_2 = trainNetwork(combined_Datastore_Train,lgraph,options)
            save UNet_2X_ICA_model_2
        elseif Mod == 3
            xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
            UNet_2X_ICA_model_3 = trainNetwork(combined_Datastore_Train,lgraph,options);
            save UNet_2X_ICA_model_3
        elseif Mod == 4            
            xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
            UNet_2X_ICA_model_4 = trainNetwork(combined_Datastore_Train,lgraph,options);
            save UNet_2X_ICA_model_4
        elseif Mod == 5             
            xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
            UNet_2X_ICA_model_5 = trainNetwork(combined_Datastore_Train,lgraph,options);
            save UNet_2X_ICA_model_5
%         elseif Mod == 6            
%             xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
%             UNet_2X_ICA_model_6 = trainNetwork(combined_Datastore_Train,lgraph,options);
%             save UNet_2X_ICA_model_6
%         elseif Mod == 7             
%             xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
%             UNet_2X_ICA_model_7 = trainNetwork(combined_Datastore_Train,lgraph,options);
%             save UNet_2X_ICA_model_7
%         elseif Mod == 8             
%             xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
%             UNet_2X_ICA_model_8 = trainNetwork(combined_Datastore_Train,lgraph,options);
%             save UNet_2X_ICA_model_8
%         elseif Mod == 9             
%             xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
%             UNet_2X_ICA_model_9 = trainNetwork(combined_Datastore_Train,lgraph,options);
%             save UNet_2X_ICA_model_9
%         elseif Mod ==10             
%             xlswrite('Test_Img_files.xlsx',Test_files,sprintf('Sheet%d',Mod),'A1');
%             UNet_2X_ICA_model_10 = trainNetwork(combined_Datastore_Train,lgraph,options);
%             save UNet_2X_ICA_model_10

    end
end
sprintf('Training-Complete');
