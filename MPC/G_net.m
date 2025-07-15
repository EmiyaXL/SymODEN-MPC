function G_u = G_net(Gw_3,Gw_2,Gw_1,Gb_3,Gb_2,Gb_1,zq,u)
gq=Gw_3*tanh(Gw_2*tanh(Gw_1*zq+Gb_1)+Gb_2)+Gb_3;
gu=gq*u;
zero_vec=zeros(3,1);
G_u=[zero_vec;gu];
end

