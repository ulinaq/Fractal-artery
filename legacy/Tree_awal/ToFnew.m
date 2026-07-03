A = imread('frog_domain_low.png');
B = im2bw(A);
Bz = size(A);
aa = 1-B;
aa = reshape(aa,prod(Bz),1);


P = ones(size(B))*max(max(Pnew));
P(B) = Pnew(B);
V = Flux(Pnew,Grid,Trans,q,aa);
Ti = reshape(V.Ti,Grid.Nx,Grid.Ny);
Tin = ones(size(Ti))*max(V.Ti(V.Ti>1));
Tin(B) = Ti(B);
figure()
imagesc(Tin); colormap(gray); title('ToF');
% contourf(P); colormap(gray); colorbar; title('Pressure');
hcb=colorbar
% title(hcb,'[kPa]')
xlabel('x [\times 3.2\cdot10^{-2} mm]')
ylabel('y [\times 3.2\cdot10^{-2} mm]')
hold on
plot(GV,'b','YData',vnode.XData,'XData',vnode.YData,'LineWidth',GV.Edges.width);
daspect([1 1 1])
plot(GA,'r','YData',node.XData,'XData',node.YData,'LineWidth',GA.Edges.width);
daspect([1 1 1])
