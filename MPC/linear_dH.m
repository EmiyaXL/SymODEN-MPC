function l_H = linear_dH(Hw_1,Hw_2,Hw_3,Hb_1,Hb_2,Hb_3,x0,x,input_H)
H_shape=input_H;
s1=Hw_1*x0+Hb_1;
s2=tanh(s1);
s3=Hw_2*s2+Hb_2;
s4=tanh(s3);
H=Hw_3*s4+Hb_3;
dHds4=Hw_3;
ds4ds3=diag(1-power(tanh(s3),2));
ds3ds2=Hw_2;
ds2ds1=diag(1-power(tanh(s1),2));
ds1dx=Hw_1;
dHdx=dHds4*ds4ds3*ds3ds2*ds2ds1*ds1dx;
A=zeros(6,6);
for n=1:6 %原求导向量维度
    for i=1:6 %对状态量每一个分量求导，最终维度为n*i
        f=zeros(1,H_shape);
        df=zeros(H_shape,6);
        g=zeros(1,H_shape);
        dg=zeros(1,H_shape);
        h=zeros(1,H_shape);
        dh=zeros(1,H_shape);
        I=zeros(1,H_shape);
        dI=zeros(1,H_shape);
        C=zeros(1,H_shape);
        dC=zeros(1,H_shape);
        a=zeros(1,H_shape);
        da=zeros(8,H_shape);
        b=zeros(1,H_shape);
        db=zeros(1,H_shape);
        d1=zeros(1,6);
        d2=zeros(1,6);
        for m=1:H_shape
            hsum=0;
            dhsum=0;
            csum=0;
            dcsum=0;
            a_sum=0;
            for j=1:H_shape
                f(j)=s1(j);
                df(j,i)=Hw_1(j,i);
                g(j)=tanh(f(j));
                dg(j)=1-power(tanh(f(j)),2);
                dhsum=dhsum+Hw_2(m,j)*dg(j)*df(j,i);
                hsum=hsum+Hw_2(m,j)*g(j);
                for k=1:6
                    a_sum=a_sum+Hw_1(j,k)*x0(k);
                end
                a(j)=a_sum+Hb_1(j);
                da(j,i)=Hw_1(j,i);
                b(j)=1-power(tanh(a(j)),2);
                db(j)=-2*tanh(a(j))*(1-power(tanh(a(j)),2));
                csum=csum+b(j)*Hw_1(j,n)*Hw_2(m,j);
                dcsum=dcsum+db(j)*da(j)*Hw_1(j,n)*Hw_2(m,j);
            end
            C(m)=csum;
            dC(m)=dcsum;
            h(m)=hsum+Hb_2(m);
            dh(m)=dhsum+Hb_2(m);
            I(m)=1-power(tanh(h(m)),2);
            dI(m)=-2*tanh(h(m))*(1-power(tanh(h(m)),2));
            d1(i)=d1(i)+Hw_3(m)*dI(m)*dh(m)*C(m);
            d2(i)=d2(i)+I(m)*dC(m);
            A(n,i)=d1(i)+d2(i);
        end
    end
end
l_H=A*(x-x0)+dHdx';
end

