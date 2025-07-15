clc;
clear all;
close all;

load('demo');
[input_H_shape,size1]=size(Hw_2);

N=10; %预测时域长度
start1=1001;
end1=10001;

P1=end1-start1;%时间长度（仿真多少步）
[q,p,ur,mu,std] = data_arrange(px,py,phi,vx,vy,w,u_tor,start1,end1);
qp=[q p];
x0=qp(1,:);
x(:,1) = x0';
Q=eye(2)*200;
R=100;
T=0.001;

%%%***********************************************************************
input = [];
cost = [];
cost_measure(1)=0;

nx = 6; % Number of states
nu = 1; % Number of inputs

beta=10^6;

%%%***********************************************************************
for k = 1:P1 
        k
        yalmip('clear');
        u = sdpvar(nu,N);
        constraints = [];
        objective = 0;
        z1 = x(:,k);
        
        for i = 1:N
            objective = objective + (z1(1:2)'-qp(k+i-1,1:2))*Q*(z1(1:2)-qp(k+i-1,1:2)') + (u(:,i)-ur(k+i-1))'*R*(u(:,i)-ur(k+i-1)) ; %生成代价函数
            z1=Symmodel(Hw_3,Hw_2,Hw_1,Hb_3,Hb_2,Hb_1,Gw_3,Gw_2,Gw_1,Gb_3,Gb_2,Gb_1,z1(1:3),z1(4:6),qp(k+i-1,:)',u(:,i),input_H_shape,T); 
            constraints = min(ur)<=u(:,i)<=max(ur);
        end
        
        options = sdpsettings('verbose',1,'solver','ipopt');%'ipopt','fmincon','bonmin','filtersd'
        % mosekbilp                      - Solve a BILP using MOSEK
        % moseklp                        - Solve a LP using MOSEK
        % mosekmilp                      - Solve a MILP using MOSEK
        % mosekmiqcqp                    - MOSEKQCQP Solve a MIQCQP using MOSEK
        % mosekmiqp                       - Solve a MIQP using MOSEK
        % mosekqcqp                      - Solve a QCQP using MOSEK
        % mosekqp                        - Solve a QP using MOSEK
        % moseksdp

        Info=optimize(constraints,objective,options);
        
        J = value(objective);
        U=value(u(1:nu,1));

        cost(k) = J;
        cost_measure(k+1) = cost_measure(k) + x(1:2,k)'*Q*x(1:2,k) + U'*R*U;

        x(:,k+1)=Symmodel(Hw_3,Hw_2,Hw_1,Hb_3,Hb_2,Hb_1,Gw_3,Gw_2,Gw_1,Gb_3,Gb_2,Gb_1,x(1:3,k),x(4:6,k),qp(k,:)',U,input_H_shape,T);
        x(:,k+1)
        input(k) = U;
        U
end
        
        







