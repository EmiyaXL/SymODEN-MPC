function [q,p,u,mu,std] = data_arrange(px,py,phi,vx,vy,w,u_tor,start,end_f)
[px_norm,px_mu,px_std]=datanormalize(px);
[py_norm,py_mu,py_std]=datanormalize(py);
[phi_norm,phi_mu,phi_std]=datanormalize(phi);
[ix_norm,ix_mu,ix_std]=datanormalize(vx);
[iy_norm,iy_mu,iy_std]=datanormalize(vy);
[iw_norm,iw_mu,iw_std]=datanormalize(w);
[u,u_mu,u_std]=datanormalize(u_tor);
q=[px_norm(start:end_f) py_norm(start:end_f) phi_norm(start:end_f)];
p=[ix_norm(start:end_f) iy_norm(start:end_f) iw_norm(start:end_f)];
u=u(start:end_f);
mu=[px_mu,py_mu phi_mu ix_mu iy_mu iw_mu u_mu];
std=[px_std py_std phi_std ix_std iy_std iw_std u_std];
end