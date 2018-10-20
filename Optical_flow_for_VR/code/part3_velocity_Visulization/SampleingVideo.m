function [num]=SampleingVideo(fileName, sampleInterval)
    
    obj = VideoReader(fileName);
    numFrames = obj.NumberOfFrames;% 帧的总数
    for k = 1 :sampleInterval: numFrames
        frame = read(obj,k);
         frame=rgb2gray(frame);
         frame=im2double(frame);
         frame1 = imresize(frame, 0.5);
    % figure
    % imshow(Frame);%
        if ~exist('win64','dir')==0
            mkdir('Frames');
        end
        imwrite(frame1,strcat('Frames\',strcat(num2str(k),'.jpg')),'jpg');% 保存帧
        num = floor(numFrames/sampleInterval);
    end
end
    