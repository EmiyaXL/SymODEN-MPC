function [norm,mu,theta] = datanormalize(data)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
mu=mean(data);
theta=std(data);
dim=size(data);
norm=zeros(dim(1),dim(2));
for i=1:dim(1)
    norm(i)=(data(i)-mu)/theta;
end
end

