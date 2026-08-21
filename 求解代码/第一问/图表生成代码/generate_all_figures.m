function generate_all_figures(whichFigure)
%GENERATE_ALL_FIGURES 2024B 第一问五张论文图的一键生成程序
%   generate_all_figures          生成全部图
%   generate_all_figures('fig3')  仅生成图3

% 数据均从 ../data 读取，图片统一输出到 ../figures。

if nargin < 1
    whichFigure = 'all';
end
close all force;
set(groot,'DefaultTextFontName','SimHei', ...
    'DefaultAxesFontName','SimHei', ...
    'DefaultAxesFontSize',11, ...
    'DefaultTextFontSize',11, ...
    'DefaultTextInterpreter','tex', ...
    'DefaultLegendInterpreter','tex', ...
    'DefaultAxesTickLabelInterpreter','tex');

codeDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(codeDir);
dataDir = fullfile(rootDir,'data');
outDir = fullfile(rootDir,'figures');
if ~isfolder(outDir), mkdir(outDir); end

C = palette();
jobs = cell(1,6);
jobs{2} = @() fig2_joint(dataDir,outDir,C);
jobs{3} = @() fig3_regions(dataDir,outDir,C);
jobs{4} = @() fig4_performance(dataDir,outDir,C);
jobs{5} = @() fig5_risk(dataDir,outDir,C);
jobs{6} = @() fig6_sensitivity(dataDir,outDir,C);

if strcmpi(whichFigure,'all')
    ids = 2:6;
else
    tok = regexp(lower(string(whichFigure)),'[2-6]','match','once');
    assert(~isempty(tok),'参数必须为 all 或 fig2 至 fig6。');
    ids = str2double(tok);
end

for i = ids
    fprintf('正在生成图%d...\n',i);
    jobs{i}();
end
fprintf('完成。图片目录：%s\n',outDir);
end

function C = palette()
nature = [110 143 178;125 164 148;234 182 122;229 167 154; ...
          193 110 113;171 200 229;216 160 193;159 141 184; ...
          208 208 138]/255;
C.blue=nature(1,:); C.green=nature(2,:); C.gold=nature(3,:);
C.salmon=nature(4,:); C.red=nature(5,:); C.lightBlue=nature(6,:);
C.pink=nature(7,:); C.purple=nature(8,:); C.olive=nature(9,:);
C.ink=[0.20 0.22 0.24]; C.gray=[0.67 0.69 0.71];
C.pale=[0.95 0.96 0.97];
C.blueGreen=interp1([0;1],[C.blue;C.green],linspace(0,1,256));
% 多节点紫金渐变，供稀疏三角曲面连续插值着色
gradientStops=[0 .22 .48 .73 1];
gradientColors=[63 55 105; 116 86 150; 175 129 169; 224 168 143; 245 202 105]/255;
C.purpleGold=interp1(gradientStops,gradientColors,linspace(0,1,256),'linear');
end

function styleAxes(ax)
set(ax,'Color','w','Box','on','LineWidth',0.9,'TickDir','out', ...
    'GridAlpha',0.18,'GridLineStyle','--','MinorGridAlpha',0.08, ...
    'Layer','top','FontName','SimHei');
end

function saveFigure(fig,outDir,fileName)
drawnow;
exportgraphics(fig,fullfile(outDir,fileName),'Resolution',300,'BackgroundColor','w');
close(fig);
end

function fig2_joint(dataDir,outDir,C)
F=readtable(fullfile(dataDir,'q1_exact_binomial_boundaries.csv'));
S=readtable(fullfile(dataDir,'q1_sequential_boundaries.csv'));
F=F(F.n<=500,:); S=S(S.n<=500,:);
[~,iF,iS]=intersect(F.n,S.n); F=F(iF,:); S=S(iS,:);
fA=F.accept_boundary./F.n; sA=S.accept_boundary./S.n;
fR=F.reject_boundary./F.n; sR=S.reject_boundary./S.n;

fig=figure('Color','w','Position',[45 45 1280 760]);
axA=axes(fig,'Position',[.070 .475 .405 .365]);
adA=axes(fig,'Position',[.070 .115 .405 .255]);
axR=axes(fig,'Position',[.555 .475 .405 .365]);
adR=axes(fig,'Position',[.555 .115 .405 .255]);
plotBoundaryComparison(axA,adA,F.n,fA,sA,'(a) 接收边界轨迹',C,true);
plotBoundaryComparison(axR,adR,F.n,fR,sR,'(b) 拒收边界轨迹',C,false);
sgtitle(fig,'固定样本与序贯规则：边界轨迹及偏移诊断', ...
    'FontSize',16,'FontWeight','bold');
saveFigure(fig,outDir,'图2_固定样本与序贯停止边界对比.png');
end

function plotBoundaryComparison(ax,ad,n,fixedRate,seqRate,panelTitle,C,showLegend)
ok=isfinite(fixedRate)&isfinite(seqRate);
n=n(ok); fixedRate=fixedRate(ok); seqRate=seqRate(ok);

hold(ax,'on'); grid(ax,'on'); styleAxes(ax);
% 标称次品率邻域、两规则间距和代表节点共同构成主视觉层次
patch(ax,[n(1) n(end) n(end) n(1)],[.09 .09 .11 .11],C.gold, ...
    'FaceAlpha',.12,'EdgeColor','none','HandleVisibility','off');
gap=patch(ax,[n;flipud(n)],[fixedRate;flipud(seqRate)],C.gold, ...
    'FaceAlpha',.20,'EdgeColor','none','HandleVisibility','off'); %#ok<NASGU>
hF=plot(ax,n,fixedRate,'--','Color',darken(C.blue,.12),'LineWidth',1.9);
hS=plot(ax,n,seqRate,'-','Color',darken(C.green,.18),'LineWidth',2.35);
yline(ax,.10,':','p_0 = 0.10','Color',C.ink,'LineWidth',1.15, ...
    'LabelHorizontalAlignment','left','LabelVerticalAlignment','top');

nodes=[100 250 500];
for j=1:numel(nodes)
    k=find(n==nodes(j),1);
    if isempty(k), continue; end
    plot(ax,[n(k) n(k)],[fixedRate(k) seqRate(k)],'-','Color',mix(C.gold,C.ink,.18),'LineWidth',1.1, ...
        'HandleVisibility','off');
    scatter(ax,n(k),fixedRate(k),38,C.blue,'s','filled','MarkerEdgeColor','w','LineWidth',.6, ...
        'HandleVisibility','off');
    scatter(ax,n(k),seqRate(k),42,C.green,'o','filled','MarkerEdgeColor','w','LineWidth',.6, ...
        'HandleVisibility','off');
end
xlim(ax,[1 500]);
allY=[fixedRate;seqRate;.09;.11]; span=max(allY)-min(allY);
ylim(ax,[max(0,min(allY)-.07*span) min(1.02,max(allY)+.08*span)]);
ticks=yticks(ax); yticklabels(ax,compose('%.0f%%',100*ticks));
ylabel(ax,'边界次品率'); title(ax,panelTitle,'FontSize',13,'FontWeight','bold');
set(ax,'XTickLabel',[]);
if showLegend
    legend(ax,[hF hS],{'固定样本规则','序贯停止规则'},'Location','northeast', ...
        'NumColumns',2,'FontSize',9.5);
end
text(ax,.985,.055,'浅金色：两规则边界间距','Units','normalized', ...
    'HorizontalAlignment','right','Color',darken(C.gold,.38),'FontSize',8.8);

hold(ad,'on'); grid(ad,'on'); styleAxes(ad);
d=100*(seqRate-fixedRate); % 百分点
dPos=max(d,0); dNeg=min(d,0);
area(ad,n,dPos,'FaceColor',mix(C.gold,[1 1 1],.12),'FaceAlpha',.55, ...
    'EdgeColor','none','HandleVisibility','off');
area(ad,n,dNeg,'FaceColor',mix(C.purple,[1 1 1],.18),'FaceAlpha',.42, ...
    'EdgeColor','none','HandleVisibility','off');
plot(ad,n,d,'Color',mix(C.ink,C.blue,.20),'LineWidth',.70,'HandleVisibility','off');
hM=plot(ad,n,movmean(d,21),'Color',darken(C.purple,.25),'LineWidth',2.0);
yline(ad,0,'-','Color',C.ink,'LineWidth',1.0,'HandleVisibility','off');
step=max(1,ceil(numel(n)/70)); q=1:step:numel(n);
scatter(ad,n(q),d(q),15,mix(C.blue,[1 1 1],.18),'filled', ...
    'MarkerFaceAlpha',.48,'MarkerEdgeColor','none','HandleVisibility','off');

[maxDev,im]=max(abs(d));
scatter(ad,n(im),d(im),54,C.red,'d','filled','MarkerEdgeColor','w','LineWidth',.7, ...
    'HandleVisibility','off');
text(ad,n(im),d(im),sprintf('  最大 %.2f pp  (n=%d)',maxDev,n(im)), ...
    'Color',darken(C.red,.20),'FontSize',8.8,'FontWeight','bold', ...
    'VerticalAlignment','bottom','Clipping','on');
lim=max(abs(d))*1.18; if lim<.15, lim=.15; end
xlim(ad,[1 500]); ylim(ad,[-lim lim]);
xlabel(ad,'累计样本量 n'); ylabel(ad,'序贯 - 固定（百分点）');
title(ad,'边界偏移诊断','FontSize',10.5,'FontWeight','bold');
legend(ad,hM,{'21 节点移动均值'},'Location','southeast','FontSize',8.5);
stats={sprintf('|偏移|均值  %.2f pp',mean(abs(d))); ...
       sprintf('末端偏移  %.2f pp',d(end))};
if showLegend, tx=.018; align='left'; else, tx=.982; align='right'; end
text(ad,tx,.91,stats,'Units','normalized','VerticalAlignment','top', ...
    'HorizontalAlignment',align, ...
    'Color',C.ink,'FontSize',8.8,'BackgroundColor',[1 1 1], ...
    'EdgeColor',mix(C.gray,[1 1 1],.38),'Margin',4);
end

function fig3_regions(dataDir,outDir,C)
S=readtable(fullfile(dataDir,'q1_sequential_boundaries.csv'));
S=S(S.n<=500,:); n=S.n; a=S.accept_boundary_rate; r=S.reject_boundary_rate;
va=isfinite(a); vr=isfinite(r); vb=va&vr;
fig=figure('Color','w','Position',[60 60 1220 720]);
ax=axes(fig,'Position',[.075 .12 .70 .78]); hold(ax,'on'); grid(ax,'on'); styleAxes(ax);
patch(ax,[n(va);flipud(n(va))],[zeros(sum(va),1);flipud(a(va))],C.green,'FaceAlpha',.24,'EdgeColor','none');
patch(ax,[n(vb);flipud(n(vb))],[a(vb);flipud(r(vb))],C.lightBlue,'FaceAlpha',.31,'EdgeColor','none');
patch(ax,[n(vr);flipud(n(vr))],[r(vr);ones(sum(vr),1)],C.salmon,'FaceAlpha',.20,'EdgeColor','none');
hA=stairs(ax,n,a,'-','Color',darken(C.green,.25),'LineWidth',2.2);
hR=stairs(ax,n,r,'--','Color',C.red,'LineWidth',2.2);
h0=yline(ax,.10,'-.','p_0 = 0.10','Color',C.ink,'LineWidth',1.4,'LabelHorizontalAlignment','left');
xline(ax,100,':','n = 100','Color',C.gray,'LineWidth',1.0,'LabelVerticalAlignment','bottom');
i100=find(n==100,1); plot(ax,100,a(i100),'o','MarkerFaceColor',C.green,'MarkerEdgeColor','k');
plot(ax,100,r(i100),'s','MarkerFaceColor',C.red,'MarkerEdgeColor','k');
text(ax,112,a(i100)-.008,sprintf('接收边界 %.2f',a(i100)),'Color',darken(C.green,.3),'FontWeight','bold');
text(ax,112,r(i100)+.012,sprintf('拒收边界 %.2f',r(i100)),'Color',C.red,'FontWeight','bold');
plot(ax,34,0,'o','MarkerSize',7,'MarkerFaceColor',C.green,'MarkerEdgeColor','k');
text(ax,47,.012,'最早接收 n = 34，k = 0','Color',darken(C.green,.3),'FontWeight','bold');
xlim(ax,[1 500]); ylim(ax,[0 .36]); xlabel(ax,'累计样本量 n'); ylabel(ax,'累计样本次品率 k/n');
title(ax,'序贯停止边界与三决策区域','FontSize',16,'FontWeight','bold');
legend(ax,[hA hR h0],{'接收边界 a_n/n','拒收边界 r_n/n','标称次品率'},'Location','northeast','FontSize',10);
text(ax,.82,.78,'拒收区','Units','normalized','Color',C.red,'FontSize',12,'FontWeight','bold');
text(ax,.82,.40,'继续抽样区','Units','normalized','Color',darken(C.blue,.2),'FontSize',12,'FontWeight','bold');
text(ax,.82,.055,'接收区','Units','normalized','Color',darken(C.green,.3),'FontSize',12,'FontWeight','bold');

ai=axes(fig,'Position',[.80 .52 .175 .32]); hold(ai,'on'); grid(ai,'on'); styleAxes(ai);
q=n<=60; qai=q&va; qri=q&vr; qbi=q&vb;
patch(ai,[n(qai);flipud(n(qai))],[zeros(sum(qai),1);flipud(a(qai))],C.green,'FaceAlpha',.24,'EdgeColor','none');
patch(ai,[n(qbi);flipud(n(qbi))],[a(qbi);flipud(r(qbi))],C.lightBlue,'FaceAlpha',.30,'EdgeColor','none');
patch(ai,[n(qri);flipud(n(qri))],[r(qri);ones(sum(qri),1)],C.salmon,'FaceAlpha',.18,'EdgeColor','none');
stairs(ai,n(q),a(q),'-','Color',darken(C.green,.25),'LineWidth',1.5);
stairs(ai,n(q),r(q),'--','Color',C.red,'LineWidth',1.5); yline(ai,.1,'-.','Color',C.ink);
plot(ai,2,1,'s','MarkerFaceColor',C.red,'MarkerEdgeColor','k','MarkerSize',6);
plot(ai,34,0,'o','MarkerFaceColor',C.green,'MarkerEdgeColor','k','MarkerSize',6);
text(ai,5,.89,'最早拒收\newline n = 2，k = 2','Color',C.red,'FontSize',9.5,'FontWeight','bold');
xlim(ai,[1 60]); ylim(ai,[0 1.02]); title(ai,'早期边界放大','FontSize',11.5,'FontWeight','bold');
xlabel(ai,'n'); ylabel(ai,'k/n');
saveFigure(fig,outDir,'图3_序贯停止边界与三决策区域.png');
end

function fig4_performance(dataDir,outDir,C)
T=readtable(fullfile(dataDir,'q1_sensitivity_true_p.csv'));
p=T.true_p; prob=[T.accept_prob T.unresolved_prob T.reject_prob]; cost=T.truncated_expected_n;
sizes=42+300*sqrt(T.unresolved_prob./max(T.unresolved_prob));
fig=figure('Color','w','Position',[80 40 1120 900]);
tl=tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');
ax1=nexttile(tl); hold(ax1,'on'); grid(ax1,'on'); styleAxes(ax1);
A=area(ax1,p,prob,'LineWidth',.8);
A(1).FaceColor=C.green; A(1).FaceAlpha=.72; A(1).EdgeColor=darken(C.green,.2);
A(2).FaceColor=C.lightBlue; A(2).FaceAlpha=.72; A(2).EdgeColor=C.blue;
A(3).FaceColor=C.salmon; A(3).FaceAlpha=.68; A(3).EdgeColor=C.red;
xline(ax1,.10,'-.','Color',C.ink,'LineWidth',1.4);
text(ax1,.102,.88,'p_0 = 0.10','Color',C.ink,'FontWeight','bold');
ylim(ax1,[0 1]); xlim(ax1,[min(p) max(p)]); ylabel(ax1,'决策概率');
title(ax1,'a 决策概率构成','FontSize',13.5,'FontWeight','bold');
legend(ax1,{'接收概率','未决概率','拒收概率'},'Location','eastoutside');
text(ax1,.03,.93,'P_A + P_U + P_R = 1','Units','normalized','FontWeight','bold','Color',C.ink);

ax2=nexttile(tl); hold(ax2,'on'); grid(ax2,'on'); styleAxes(ax2);
yl=[0 max(cost)*1.08]; patch(ax2,[.09 .11 .11 .09],[yl(1) yl(1) yl(2) yl(2)],C.gold,'FaceAlpha',.14,'EdgeColor','none');
hL=plot(ax2,p,cost,'-','Color',C.blue,'LineWidth',2.2);
hB=scatter(ax2,p,cost,sizes,T.distance_to_p0,'filled','MarkerFaceAlpha',.78,'MarkerEdgeColor',C.ink,'LineWidth',.5);
colormap(ax2,C.blueGreen); cb=colorbar(ax2); cb.Color=C.ink; cb.Label.String='到标称值的距离 |p-p_0|'; cb.Label.FontName='SimHei';
xline(ax2,.10,'-.','Color',C.ink,'LineWidth',1.4);
text(ax2,.102,yl(2)*.86,'p_0 = 0.10','Color',C.ink,'FontWeight','bold');
[peak,ip]=max(cost); plot(ax2,p(ip),peak,'p','MarkerSize',13,'MarkerFaceColor',C.gold,'MarkerEdgeColor',C.ink);
text(ax2,p(ip)+.006,peak*.93,sprintf('成本峰值 %.2f\n未决概率 %.2f',peak,T.unresolved_prob(ip)), ...
    'Color',C.ink,'FontWeight','bold','BackgroundColor','w','Margin',3);
xlim(ax2,[min(p) max(p)]); ylim(ax2,yl); xlabel(ax2,'真实次品率 p'); ylabel(ax2,'截尾期望检测样本量');
title(ax2,'b 检测成本变化','FontSize',13.5,'FontWeight','bold');
legend(ax2,[hL hB],{'截尾期望检测样本量','气泡大小表示未决概率'},'Location','eastoutside');
title(tl,'真实次品率对检验性能与检测成本的影响','FontSize',16,'FontWeight','bold');
saveFigure(fig,outDir,'图4_真实性能与检测成本.png');
end

function fig5_risk(dataDir,outDir,C)
R=readtable(fullfile(dataDir,'q1_risk_grid_validation.csv'));
WR=R(strcmp(string(R.risk_type),'wrong_reject'),:); WA=R(strcmp(string(R.risk_type),'wrong_accept'),:);
mx=max(R.unresolved_prob); sR=52+290*sqrt(WR.unresolved_prob./mx); sA=52+290*sqrt(WA.unresolved_prob./mx);
fig=figure('Color','w','Position',[80 70 1120 690]);
ax=axes(fig,'Position',[.09 .13 .72 .76]); hold(ax,'on'); grid(ax,'on'); styleAxes(ax);
patch(ax,[.01 .10 .10 .01],[0 0 .105 .105],C.green,'FaceAlpha',.07,'EdgeColor','none');
patch(ax,[.10 .19 .19 .10],[0 0 .105 .105],C.blue,'FaceAlpha',.07,'EdgeColor','none');
h5=plot(ax,[.01 .10],[.05 .05],'--','Color',C.red,'LineWidth',1.7);
h10=plot(ax,[.10 .19],[.10 .10],'-.','Color',C.blue,'LineWidth',1.7);
hWR=plot(ax,WR.true_p,WR.finite_horizon_risk,'-o','Color',C.red,'MarkerFaceColor','w','LineWidth',1.9,'MarkerSize',5);
hWA=plot(ax,WA.true_p,WA.finite_horizon_risk,'--s','Color',C.blue,'MarkerFaceColor','w','LineWidth',1.9,'MarkerSize',5);
scatter(ax,WR.true_p,WR.finite_horizon_risk,sR,C.red,'o','filled','MarkerFaceAlpha',.60,'MarkerEdgeColor','k');
scatter(ax,WA.true_p,WA.finite_horizon_risk,sA,C.blue,'s','filled','MarkerFaceAlpha',.60,'MarkerEdgeColor','k');
xline(ax,.10,':','Color',C.ink,'LineWidth',1.4);
text(ax,.505,.93,'p_0 = 0.10','Units','normalized','Color',C.ink,'FontWeight','bold');
[mR,iR]=max(WR.finite_horizon_risk); [mA,iA]=max(WA.finite_horizon_risk);
plot(ax,WR.true_p(iR),mR,'p','MarkerSize',12,'MarkerFaceColor',C.gold,'MarkerEdgeColor','k');
plot(ax,WA.true_p(iA),mA,'p','MarkerSize',12,'MarkerFaceColor',C.gold,'MarkerEdgeColor','k');
text(ax,WR.true_p(iR)-.046,mR+.006,sprintf('错误拒收风险最高 %.3f\n低于上限 0.05',mR),'Color',C.red,'FontWeight','bold','BackgroundColor','w','Margin',3);
text(ax,WA.true_p(iA)+.007,mA+.006,sprintf('错误接收风险最高 %.3f\n低于上限 0.10',mA),'Color',C.blue,'FontWeight','bold','BackgroundColor','w','Margin',3);
text(ax,.02,.94,'气泡越大表示截尾时未决概率越高','Units','normalized','FontSize',10.5,'Color',C.ink);
xlim(ax,[.01 .19]); ylim(ax,[0 .108]); xlabel(ax,'真实次品率 p'); ylabel(ax,'错误风险');
title(ax,'序贯检验错误风险控制验证','FontSize',16,'FontWeight','bold');
legend(ax,[hWR hWA h5 h10],{'错误拒收风险','错误接收风险','错误拒收上限 5%','错误接收上限 10%'}, ...
    'Location','eastoutside','FontSize',10);
saveFigure(fig,outDir,'图5_序贯检验错误风险控制验证.png');
end

function fig6_sensitivity(dataDir,outDir,C)
T=readtable(fullfile(dataDir,'q1_sensitivity_confidence.csv'));
A=T(strcmp(string(T.varied_parameter),'accept_confidence'),:);
B=T(strcmp(string(T.varied_parameter),'reject_confidence'),:);
fig=figure('Color','w','Position',[25 55 1500 720]);
ax1=axes(fig,'Position',[.055 .11 .34 .75]);
ax2=axes(fig,'Position',[.555 .11 .34 .75]);
plotTri(ax1,A,'accept_confidence','接收置信水平','a 接收侧置信水平',C);
plotTri(ax2,B,'reject_confidence','拒收置信水平','b 拒收侧置信水平',C);
annotation(fig,'textbox',[.12 .935 .76 .05], ...
    'String','真实次品率与置信水平对检测成本的联合影响', ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'EdgeColor','none','FontName','SimHei','FontSize',16,'FontWeight','bold');
saveFigure(fig,outDir,'图6_置信水平敏感性三角曲面.png');
end

function plotTri(ax,T,cName,yLabelText,titleText,C)
hold(ax,'on'); grid(ax,'on'); styleAxes(ax);
x=T.true_p; y=T.(cName); z=T.truncated_expected_n;
tri=delaunay(x,y);
hSurf=trisurf(tri,x,y,z,z,'Parent',ax, ...
    'FaceColor','interp','FaceAlpha',.97, ...
    'EdgeColor',[.40 .38 .46],'EdgeAlpha',.18,'LineWidth',.35, ...
    'FaceLighting','gouraud','AmbientStrength',.72, ...
    'DiffuseStrength',.58,'SpecularStrength',.06,'SpecularExponent',8);
scatter3(ax,x,y,z,23,'w','filled','MarkerFaceAlpha',.78, ...
    'MarkerEdgeColor',[.30 .30 .32],'LineWidth',.4);
colormap(ax,C.purpleGold); cb=colorbar(ax); cb.Color=C.ink; cb.Label.String='截尾期望检测样本量'; cb.Label.FontName='SimHei';
xlabel(ax,'真实次品率 p'); ylabel(ax,yLabelText); zlabel(ax,'截尾期望检测样本量');
title(ax,titleText,'FontSize',13.5,'FontWeight','bold');
view(ax,39,29); axis(ax,'tight'); axis(ax,'vis3d');
pbaspect(ax,[1.18 1 .66]); set(ax,'Projection','perspective');
lighting(ax,'gouraud'); camlight(ax,'headlight');

sens=confidenceSensitivity(T,cName);
[~,ord]=sort(sens,'descend','MissingPlacement','last'); ord=ord(isfinite(sens(ord)));
chosen=[];
for ii=1:numel(ord)
    q=ord(ii);
    if isempty(chosen) || all(abs(x(q)-x(chosen))>.005 | abs(y(q)-y(chosen))>.005)
        chosen(end+1)=q; %#ok<AGROW>
    end
    if numel(chosen)==2, break; end
end
for k=1:numel(chosen)
    q=chosen(k);
    scatter3(ax,x(q),y(q),z(q),95,C.red,'filled','MarkerEdgeColor','k','LineWidth',.8);
end
summary=sprintf('梯度敏感点\n1  p = %.2f，C = %.2f\n2  p = %.2f，C = %.2f', ...
    x(chosen(1)),y(chosen(1)),x(chosen(2)),y(chosen(2)));
text(ax,.03,.94,summary,'Units','normalized','Color',C.red,'FontSize',9.5, ...
    'FontWeight','bold','VerticalAlignment','top','BackgroundColor','w','Margin',3);
end

function s=confidenceSensitivity(T,cName)
s=nan(height(T),1); ps=unique(T.true_p);
for i=1:numel(ps)
    idx=find(abs(T.true_p-ps(i))<1e-12);
    [c,ord]=sort(T.(cName)(idx)); q=idx(ord); z=T.truncated_expected_n(q);
    if numel(c)>1
        g=abs(gradient(z)./gradient(c)); s(q)=g;
    end
end
end

function out=mix(a,b,w)
out=(1-w)*a+w*b;
end

function out=darken(a,w)
out=(1-w)*a;
end
