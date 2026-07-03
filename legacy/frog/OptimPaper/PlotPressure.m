function yr = PlotPressure(prm,segment,vsegment,node,x,name)
nam = ['Fig/PressureDist',name];
figure();
for i = 1:segment.n
    y1(i) = mean(x([segment.end(i),segment.init(i)]))*prm.mmHg;
end
x1 = 1000*segment.rad;

for i = 1:vsegment.n
    y2(i) = mean(x(node.n+[vsegment.end(i),vsegment.init(i)]))*prm.mmHg;
end
x2 = vsegment.rad*1000;

ax(1) = axes('position', [0.1 0.1 0.4 0.8], 'box', 'off', 'yaxisloc', 'left', 'xdir', 'reverse');
ax(2) = axes('position', [0.5 0.1 0.4 0.8], 'box', 'off', 'yaxisloc', 'right');
set(ax,'ylim', [0 40]);
set(ax(1),'xlim', [0 160]);
set(ax(2),'xlim', [0 220]);
ax(3) = copyobj(ax(1), gcf);
ax(4) = copyobj(ax(2), gcf);
set(ax(3:4), 'ycolor', 'none', 'xaxisloc', 'top', 'xticklabel', '', 'color', 'none');
hold(ax(1), 'on');
hold(ax(2), 'on');
plot(ax(1), x1,y1,'r*');title(name);
plot(ax(2), x2,y2,'b*');
ax(2).YAxis.Visible = 'off';
xlabel(ax(1),['Radius of vessel [\mu','m]'])
ylabel(ax(1),'Pressure [mmHg]')
set(gcf, 'Color', 'w'); 
export_fig(nam,'-r300'); 
savefig(nam);
yr = 0;
for i = 1:vsegment.n
    if vsegment.rad(i)<0.032
    yr(end+1) = mean(x(node.n+[vsegment.end(i),vsegment.init(i)]))*prm.mmHg;
    end
end
yr = 0;
for i = 1:segment.n
    if segment.rad(i)<0.032
    yr(end+1) = mean(x([vsegment.end(i),vsegment.init(i)]))*prm.mmHg;
    end
end
clear ax