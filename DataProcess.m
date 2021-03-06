data = load('data.csv');

MaxOutput = prctile(abs(data(:,37:43)),99.9,1);
MinOutput = -MaxOutput;
MaxTheta = [3.14,3.14,3.14,3.14,3.14,3.14,3.14];
MinTheta = -MaxTheta;
MaxThetaDot = [0.3,0.3,0.3,0.3,0.3,0.3,0.3];
MinThetaDot = -MaxThetaDot;
csvwrite('MinMax.csv', [MaxOutput; MinOutput; MaxTheta; MinTheta; MaxThetaDot; MinThetaDot]);

TrainingData = data(1000:fix(0.7*size(data,1)),:);
ValidationData = data(fix(0.7*size(data,1)):fix(0.85*size(data,1)),:);
TestingData = data(fix(0.85*size(data,1)):end,:);

TrainingProcessed = ProcessRawData(TrainingData, MaxOutput, MinOutput, MaxTheta, MinTheta, MaxThetaDot, MinThetaDot);
TrainingDataMix = TrainingProcessed(randperm(size(TrainingProcessed,1)),:);
csvwrite('TrainingData.csv', TrainingDataMix);
clear TrainingDataMix

ValidationProcessed = ProcessRawData(ValidationData, MaxOutput, MinOutput, MaxTheta, MinTheta, MaxThetaDot, MinThetaDot);
csvwrite('ValidationData.csv', ValidationProcessed);

TestingProcessed = ProcessRawData(TestingData, MaxOutput, MinOutput, MaxTheta, MinTheta, MaxThetaDot, MinThetaDot);
csvwrite('TestingData.csv', TestingProcessed);

function ProcessedData = ProcessRawData(RawData, MaxOutput, MinOutput, MaxTheta, MinTheta, MaxThetaDot, MinThetaDot)
    data_dt = 0.01;

    num_joint = 7;
    num_feature = 2;
    num_redundant_feature = 3;
    num_sequence = 5;
    num_output = num_joint;

    Processed = zeros(size(RawData,1), num_joint*num_feature*num_sequence);
    recent_wrong_dt_idx = 0;
    DataIdx = 1;

    for k=num_sequence:size(RawData,1)
       if (round(RawData(k,1) - RawData(k-1,1),3) ~= data_dt)
            recent_wrong_dt_idx = k;
       end

       if k < recent_wrong_dt_idx + num_sequence
            continue
       end

       for joint = 1:num_output
           Processed(DataIdx, num_joint*num_feature*num_sequence + joint) = 2*(RawData(k,1+num_joint*(num_feature + num_redundant_feature)+joint) - MinOutput(joint))/(MaxOutput(joint) - MinOutput(joint))-1;
%             Processed(DataIdx, num_joint*num_feature*num_sequence + joint) = RawData(k,1+num_joint*(num_feature + num_redundant_feature)+joint);
       end

       for past_time_step = 1:num_sequence
           for joint = 1:num_joint
               Processed(DataIdx, (num_sequence-past_time_step)*num_joint*num_feature + (joint-1)*num_feature + 1) = 2*(RawData(k-past_time_step+1,1+joint) - MinTheta(joint))/(MaxTheta(joint) - MinTheta(joint))-1;
               Processed(DataIdx, (num_sequence-past_time_step)*num_joint*num_feature + (joint-1)*num_feature + 2) = 2*(RawData(k-past_time_step+1,8+joint) - MinThetaDot(joint))/(MaxThetaDot(joint) - MinThetaDot(joint))-1;
           end
       end

       DataIdx = DataIdx + 1;
    end
    ProcessedData = Processed(1:DataIdx-1,:);
end