function z = Synmodel(Hw_3,Hw_2,Hw_1,Hb_3,Hb_2,Hb_1,Gw_3,Gw_2,Gw_1,Gb_3,Gb_2,Gb_1,zq,zp,tq,u,h_shape,sample_time)
q_p=[zq;zp];
%[Hv,dH]=H_gradn(q_p,Hw_3,Hw_2,Hw_1,Hb_3,Hb_2,Hb_1);
dH=linear_dH(Hw_1,Hw_2,Hw_3,Hb_1,Hb_2,Hb_3,tq,q_p,h_shape);
M=[0 0 0 1 0 0;
   0 0 0 0 1 0;
   0 0 0 0 0 1;
   -1 0 0 0 0 0;
   0 -1 0 0 0 0;
   0 0 -1 0 0 0];
H=M*dH;
G=G_net(Gw_3,Gw_2,Gw_1,Gb_3,Gb_2,Gb_1,tq(1:3),u);
z=q_p+(H+G)*sample_time;
end

