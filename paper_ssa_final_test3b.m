% Performance plot over all 80 matrices selected from the SuiteSparse
% collection.
% Performance indicator is the largest basis that can be constructed with a bounded condition number
% Used for final paper experiments
% Same setup as paper_ssa_final_test_2b.m, but uses starting vector e_1 instead of random
% Same as paper_ssa_final_test_3a.m, but starting vector is slightly perturbed


addpath('ssget')
clear all
close all hidden
mydefaults

p = 149;	% max number of Krylov iterations
s = 2;		% oversampling parameter
t = 5;		% truncation parameter
condbound = 1e12;		% upper bound on the basis condition number

index = ssget;
ids = find(index.numerical_symmetry==0 & index.ncols==index.nrows & index.ncols>=1e4 & index.ncols<1e6);
COND = []; ORTH = []; withplots = 0;
SIZE = [];	
perturb = true;		% if true, perturb starting vector (also changes filenames and figure titles)
%% 
for idj = 1:length(ids)
    id = ids(idj);
    Prob = ssget(id);
    A = Prob.A;  
    fprintf('\nPROBLEM %s (%d of %d)', Prob.name,idj,length(ids));
    
    N = size(A,1);
    rng('default');
	v0 = zeros(N,1);		v0(1) =  1;
	if perturb 
		v0 = v0 + 1e-15*ones(N, 1);
	end
    hS = srht(N, round(s*p));		% sketching matrix (from Oleg's code)
    
    %% Truncated Arnoldi:
    fprintf('\ntruncated Arnoldi\t\t\t|')
	cnd = [];
	jmax_ok = 0;
    V = []; H = [];
    V(:, 1) = v0/norm(v0);
    for j = 1:p
        if mod(j, 10) == 0, fprintf('.'); end
        if withplots, c1(j) = cond(V(:,1:j)); end

		if mod(j, 10) == 0
			cnd(j) = cond(V(:, 1:j));
			if cnd(j) > condbound
				break;
			else
				jmax_ok = j;	% last j for which cnd was below condbound
			end
		end
        w = A*V(:,j);
        H(:,j) = 0;
        cols = max(1,j-t+1):j;
        h = V(:,cols)'*w;
        H(cols,j) = h;
        w = w - V(:,cols)*h;
        H(j+1,j) = norm(w);
        V(:,j+1) = w/H(j+1,j);
    end
	jmax = size(V, 2);
	for j = jmax_ok+1:jmax
		cnd(j) = cond(V(:, 1:j));
		if cnd(j) > condbound
			SIZE(idj, 1) = j-1;
			break;
		elseif j == jmax
			SIZE(idj, 1) = j;
		end
	end

    leg{1} = 'truncated Arnoldi';
    if withplots
        figure
        semilogy(c1,'o'), hold on
    end

    %% Truncated Arnoldi sketched:
    fprintf('\ntruncated Arnoldi sketched\t\t|')
	cnd = [];
	jmax_ok = 0;
    sw = hS(v0); nsw = norm(sw);
    V = []; SV = []; SAV = []; H = [];
    SV(:,1) = sw/nsw; V(:,1) = v0/nsw;
    for j = 1:p
        if mod(j, 10) == 0, fprintf('.'); end
        if withplots, c2(j) = cond(V(:,1:j)); end
		if mod(j, 10) == 0
			cnd(j) = cond(V(:, 1:j));
			if cnd(j) > condbound
				break;
			else
				jmax_ok = j;	% last j for which cnd was below condbound
			end
		end

		w = A*V(:,j);
        sw = hS(w);
        SAV(:,j) = sw;
        H(:,j) = 0;
        cols = max(1,j-t+1):j;
        h = SV(:,cols)'*sw;
        H(cols,j) = h;
        w = w - V(:,cols)*h;
        sw = sw - SV(:,cols)*h;
        H(j+1,j) = norm(sw);
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = sw/H(j+1,j);
    end
	jmax = size(V, 2);
	for j = jmax_ok+1:jmax
		cnd(j) = cond(V(:, 1:j));
		if cnd(j) > condbound
			SIZE(idj, 2) = j-1;
			break;
		elseif j == jmax
			SIZE(idj, 2) = j;
		end
	end

	leg{2} = 'sketch + truncate';
    if withplots, semilogy(c2); end
    
    %% Sketch and select Arnoldi (pinv)
    fprintf('\nsketch and select Arnoldi (pinv)\t|')
	cnd = [];
	jmax_ok = 0;
    sw = hS(v0); nsw = norm(sw);
    V = []; SV = []; SAV = []; H = [];
    SV(:,1) = sw/nsw; V(:,1) = v0/nsw;
    for j = 1:p
        if mod(j, 10) == 0, fprintf('.'); end
        if withplots, c3(j) = cond(V(:,1:j)); end
		if mod(j, 10) == 0
			cnd(j) = cond(V(:, 1:j));
			if cnd(j) > condbound
				break;
			else
				jmax_ok = j;	% last j for which cnd was below condbound
			end
		end
        w = A*V(:,j);
        sw = hS(w);
        SAV(:,j) = sw;
        H(:,j) = 0;
        coeffs = pinv(SV(:,1:j))*sw; % instead of \

        % selection with weights
        %weights = vecnorm(SV(:,1:j)'*SV(:,1:j));
        %weights = weights(:);
        %[~,ind] = maxk(abs(coeffs).*abs(weights),t); 

        [~,ind] = maxk(abs(coeffs),t); 

        h = coeffs(ind);
        H(ind,j) = h;
        w = w - V(:,ind)*h;
        sw = sw - SV(:,ind)*h;
        H(j+1,j) = norm(sw);
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = sw/H(j+1,j);
    end
	jmax = size(V, 2);
	for j = jmax_ok+1:jmax
		cnd(j) = cond(V(:, 1:j));
		if cnd(j) > condbound
			SIZE(idj, 3) = j-1;
			break;
		elseif j == jmax
			SIZE(idj, 3) = j;
		end
	end

	leg{3} = 'sketch + select pinv';
    if withplots, semilogy(c3); end
    
    %% Sketch and select Arnoldi (pinv, recomp)
    fprintf('\nsketch and select Arnoldi (pinv, recomp)|')
	cnd = [];
	jmax_ok = 0;
    sw = hS(v0); nsw = norm(sw);
    V = []; SV = []; SAV = []; H = [];
    SV(:,1) = sw/nsw; V(:,1) = v0/nsw;
    for j = 1:p
        if mod(j, 10) == 0, fprintf('.'); end
        if withplots, c4(j) = cond(V(:,1:j)); end
		if mod(j, 10) == 0
			cnd(j) = cond(V(:, 1:j));
			if cnd(j) > condbound
				break;
			else
				jmax_ok = j;	% last j for which cnd was below condbound
			end
		end
        w = A*V(:,j);
        sw = hS(w);
        SAV(:,j) = sw;
        H(:,j) = 0;
        coeffs = pinv(SV(:,1:j))*sw;
        [~,ind] = maxk(abs(coeffs),t); 
        h = pinv(SV(:,ind))*sw; % recompute
        H(ind,j) = h;
        w = w - V(:,ind)*h;
        sw = sw - SV(:,ind)*h;
        H(j+1,j) = norm(sw);
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = sw/H(j+1,j);
	end
	jmax = size(V, 2);
	for j = jmax_ok+1:jmax
		cnd(j) = cond(V(:, 1:j));
		if cnd(j) > condbound
			SIZE(idj, 4) = j-1;
			break;
		elseif j == jmax
			SIZE(idj, 4) = j;
		end
	end

    leg{4} = 'sketch + select pinv2';
    if withplots, semilogy(c4); end

    %% Sketch and select Arnoldi (corr)
    fprintf('\nsketch and select Arnoldi (corr)\t|')
	cnd = [];
	jmax_ok = 0;
    sw = hS(v0); nsw = norm(sw);
    V = []; SV = []; SAV = []; H = [];
    SV(:,1) = sw/nsw; V(:,1) = v0/nsw;
    for j = 1:p
        if mod(j, 10) == 0, fprintf('.'); end
        if withplots, c5(j) = cond(V(:,1:j)); end
		if mod(j, 10) == 0
			cnd(j) = cond(V(:, 1:j));
			if cnd(j) > condbound
				break;
			else
				jmax_ok = j;	% last j for which cnd was below condbound
			end
		end
        w = A*V(:,j);
        sw = hS(w);
        SAV(:,j) = sw;
        H(:,j) = 0;
        coeffs = SV(:,1:j)'*sw;
        [~,ind] = maxk(abs(coeffs),t); 
        h = coeffs(ind);
        H(ind,j) = h;
        w = w - V(:,ind)*h;
        sw = sw - SV(:,ind)*h;
        H(j+1,j) = norm(sw);
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = sw/H(j+1,j);
    end
	jmax = size(V, 2);
	for j = jmax_ok+1:jmax
		cnd(j) = cond(V(:, 1:j));
		if cnd(j) > condbound
			SIZE(idj, 5) = j-1;
			break;
		elseif j == jmax
			SIZE(idj, 5) = j;
		end
	end

	leg{5} = 'sketch + select corr';
    if withplots, semilogy(c5); end
    
    %% Sketch and select Arnoldi (corr, pinv)
    fprintf('\nsketch and select Arnoldi (corr, pinv)\t|')
	cnd = [];
	jmax_ok = 0;
    sw = hS(v0); nsw = norm(sw);
    V = []; SV = []; SAV = []; H = [];
    SV(:,1) = sw/nsw; V(:,1) = v0/nsw;
    for j = 1:p
        if mod(j, 10) == 0, fprintf('.'); end
        if withplots, c6(j) = cond(V(:,1:j)); end
		if mod(j, 10) == 0
			cnd(j) = cond(V(:, 1:j));
			if cnd(j) > condbound
				break;
			else
				jmax_ok = j;	% last j for which cnd was below condbound
			end
		end
        w = A*V(:,j);
        sw = hS(w);
        SAV(:,j) = sw;
        H(:,j) = 0;
        coeffs = SV(:,1:j)'*sw;
        [~,ind] = maxk(abs(coeffs),t); 
        h = pinv(SV(:,ind))*sw; % recompute
        H(ind,j) = h;
        w = w - V(:,ind)*h;
        sw = sw - SV(:,ind)*h;
        H(j+1,j) = norm(sw);
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = sw/H(j+1,j);
    end
	jmax = size(V, 2);
	for j = jmax_ok+1:jmax
		cnd(j) = cond(V(:, 1:j));
		if cnd(j) > condbound
			SIZE(idj, 6) = j-1;
			break;
		elseif j == jmax
			SIZE(idj, 6) = j;
		end
	end

	leg{6} = 'sketch + select corr pinv';
    if withplots, semilogy(c6,'--'); end
    
	%% Orthogonal Matching Pursuit (OMP)
	fprintf('\nOrthogonal Matching Pursuit\t\t|')
	cnd = [];
	jmax_ok = 0;
	sw = hS(v0); nsw = norm(sw);
    V = []; SV = []; SAV = []; H = [];
	SV(:,1) = sw/nsw; V(:,1) = v0/nsw;
	for j = 1:p
		if mod(j, 10) == 0, fprintf('.'); end
        if withplots, c7(j) = cond(V(:,1:j)); end
		if mod(j, 10) == 0
			cnd(j) = cond(V(:, 1:j));
			if cnd(j) > condbound
				break;
			else
				jmax_ok = j;	% last j for which cnd was below condbound
			end
		end
        w = A*V(:,j);
        sw = hS(w);
		SAV(:,j) = sw;
        H(:,j) = 0;
		% OMP -- INITIALIZATION:
		r = sw;					% residual for OMP
		idx = zeros(0, 1);		% index set for orthogonalization
		SV_i = zeros(N, 0);		% selected columns of SV
		x_i = zeros(0, 1);		% to prevent errors for t = 0
		% OMP -- LOOP:
		for i = 1:min(j, t)		
            corr = abs(SV(:,1:j)'*r);			% correlations, original version
			% corr = abs(pinv(SV(:,1:j)) * r);	% alternative with pinv?
			corr(idx) = 0;					% to avoid picking the same column twice (when using variant)
			[~, idx_i] = max(corr);	% find column in SV with largest correlation with r
			idx = [idx, idx_i];				% add selected index
			SV_i = [SV_i, SV(:, idx_i)];	% add selected column
			x_i = pinv(SV_i) * sw;			% "sparse" approximation to Sw (using columns of SV_i)
			r = sw - SV_i*x_i;				% updated residual
		end		% END OMP LOOP
		% at this point we have the index set idx for orthogonalization
		% Orthogonalize and update basis (same as other methods):
        h = pinv(SV(:,idx)) * sw;		
        H(idx,j) = h;
        w = w - V(:,idx)*h;
        sw = sw - SV(:,idx)*h;
        H(j+1,j) = norm(sw);
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = sw/H(j+1,j);
	end
	jmax = size(V, 2);
	for j = jmax_ok+1:jmax
		cnd(j) = cond(V(:, 1:j));
		if cnd(j) > condbound
			SIZE(idj, 7) = j-1;
			break;
		elseif j == jmax
			SIZE(idj, 7) = j;
		end
	end

	leg{7} = 'sketch + select OMP';
    if withplots, semilogy(c7); end

	%% Subspace Pursuit (SP)
	fprintf('\nSubspace Pursuit\t\t\t|')
	cnd = [];
	jmax_ok = 0;
	itsp = 1;		% number of iterations of SP
	% SP with 0 iterations is the same as largest coeffs sketched with recomputation
	sw = hS(v0); nsw = norm(sw);
    V = []; SV = []; SAV = []; H = [];
	SV(:,1) = sw/nsw; V(:,1) = v0/nsw;
	for j = 1:p
		if mod(j, 10) == 0, fprintf('.'); end
        if withplots, c8(j) = cond(V(:,1:j)); end
		if mod(j, 10) == 0
			cnd(j) = cond(V(:, 1:j));
			if cnd(j) > condbound
				break;
			else
				jmax_ok = j;	% last j for which cnd was below condbound
			end
		end
        w = A*V(:,j);
        sw = hS(w);
		SAV(:,j) = sw;
        H(:,j) = 0;
		% SP -- INITIALIZATION:
		corr = abs(SV(:,1:j)'*sw);			% correlations, original version
		% corr = abs(pinv(SV(:,1:j)) * sw);		% alternative with pinv?
		[~, idx_i] = maxk(corr, min(j, t));	% find t columns in SV with largest correlation with r
		SV_i = SV(:, idx_i);			% select columns
		x_i = pinv(SV_i) * sw;				% corresponds to recomputing the coefficients using those columns
		Sr = sw - SV_i * x_i;			% compute residual
		% SP -- LOOP:
		for isp = 1:itsp
			y = SV' * Sr;						% correlations of basis with residual, original version
			% y = pinv(SV)*Sr;						% alternative with pinv?
			[~, idx2_i] = maxk(abs(y), t);		% find t indices with largest components
			idxU_i = union(idx_i, idx2_i);	% union of old and new index sets
			xU = pinv(SV(:, idxU_i)) * sw;
			% here, idx_rel is the index relative to the coordinates of xU!
			[~, idx_rel] = maxk(abs(xU), t);	% t (relative) indices with largest components in the union
			idx_i = idxU_i(idx_rel);			% get indices according to all columns
			SV_i = SV(:, idx_i);			% new column set
			x_i = pinv(SV_i) * sw;			% new coefficients
			Sr = sw - SV_i * x_i;			% new residual
		end		% END SP LOOP
		% at this point we have the index set idx_i for orthogonalization
		% Orthogonalize and update basis (same as other methods):
        h = pinv(SV(:,idx_i)) * sw;		
        H(idx_i,j) = h;
        w = w - V(:,idx_i)*h;
        sw = sw - SV(:,idx_i)*h;
        H(j+1,j) = norm(sw);
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = sw/H(j+1,j);
	end
	jmax = size(V, 2);
	for j = jmax_ok+1:jmax
		cnd(j) = cond(V(:, 1:j));
		if cnd(j) > condbound
			SIZE(idj, 8) = j-1;
			break;
		elseif j == jmax
			SIZE(idj, 8) = j;
		end
	end

	leg{8} = 'sketch + select SP';
    if withplots, semilogy(c8); end


	%% Sketch and select Arnoldi (greedy) - from NATARAJA paper
	fprintf('\nsketch and select Arnoldi (greedy)\t|')
	cnd = [];
	jmax_ok = 0;
	sw = hS(v0); nsw = norm(sw);
	V = []; SV = []; SAV = []; H = [];
	SV(:,1) = sw/nsw; V(:,1) = v0/nsw;
	for j = 1:p
		if mod(j, 10) == 0, fprintf('.'); end
		if withplots, c9(j) = cond(V(:,1:j)); end
		if mod(j, 10) == 0
			cnd(j) = cond(V(:, 1:j));
			if cnd(j) > condbound
				break;
			else
				jmax_ok = j;	% last j for which cnd was below condbound
			end
		end
		w = A*V(:,j);
		sw = hS(w);
		SAV(:,j) = sw;
		H(:,j) = 0;

		% get indices via greedy
		% see NATARAJA paper
		ind = [];
		SV1 = SV; sw1 = sw;
		for it = 1:min(j, t)
			corr = SV1'*sw1;
			[~,i] = max(abs(corr));
			ind = [ ind; i];
			sw1 = sw1 - SV1(:,i)*(SV1(:,i)'*sw1);
			SV1 = SV1 - SV1(:,i)*(SV1(:,i)'*SV1);
			SV1 = SV1./vecnorm(SV1);
			SV1(:,ind) = 0;
		end

		h = pinv(SV(:,ind))*sw; % recompute
		H(ind,j) = h;
		w = w - V(:,ind)*h;
		sw = sw - SV(:,ind)*h;
		H(j+1,j) = norm(sw);
		V(:,j+1) = w/H(j+1,j);
		SV(:,j+1) = sw/H(j+1,j);
	end
	jmax = size(V, 2);
	for j = jmax_ok+1:jmax
		cnd(j) = cond(V(:, 1:j));
		if cnd(j) > condbound
			SIZE(idj, 9) = j-1;
			break;
		elseif j == jmax
			SIZE(idj, 9) = j;
		end
	end
	
	leg{9} = 'sketch + select greedy';
	if withplots, semilogy(c9,':'); end


	
    if withplots 
        title(sprintf('%s', Prob.name))
        legend(leg,'Location','northwest')
    end
    
    fprintf('\n')
end


%%
figure
RELSIZE = SIZE;
for i = 1:size(SIZE, 1)
	maxsize(i) = max(SIZE(i, :));		% largest basis constructed for problem i
	RELSIZE(i, :) = maxsize(i) ./ SIZE(i, :);	
end
% RELSIZE(i, j) = 1 when method j is the best for problem i
% RELSIZE(i, j) = k > 1 means that method j constructs a basis of size maxsize(i)/k

leg2 = leg;

perfprof(RELSIZE, 2.5)
legend(leg2,'Location','southeast','NumColumns',1)
if perturb
	title(sprintf('$%d$ matrices, $m = %d$, $k = %d$, $\\textrm{cond} < 10^{%d}$, $\\textit{\\textbf{b}} = \\textit{\\textbf{e}}_1 + 10^{-15} \\textit{\\textbf{e}}$', length(ids), p+1, t, floor(log10(condbound))));
else
	title(sprintf('$%d$ matrices, $m = %d$, $k = %d$, $\\textrm{cond} < 10^{%d}$, $\\textit{\\textbf{b}} = \\textit{\\textbf{e}}_1$', length(ids), p+1, t, floor(log10(condbound))));
end
xlabel('$\theta$'), ylabel('basis size performance'); 
if perturb
	fname = sprintf('ssa_plots/test2bdim_%d_m%d_s%d_k%d_cnd%d_e1pert',length(ids),p+1,s,t,floor(log10(condbound)));
else
	fname = sprintf('ssa_plots/test2bdim_%d_m%d_s%d_k%d_cnd%d_e1',length(ids),p+1,s,t,floor(log10(condbound)));
end
% mypdf(fname,0.66,1.8)
mypdf(fname,0.66,0.85)

% plot with basis dimensions for some methods:
figure;
[~, ind] = sort(SIZE(:, 1));			% sort all based on truncated Arnoldi
% [~, ind] = sort(SIZE(:, 1));
p1 = plot(SIZE(ind, 1), 'o', 'MarkerSize', 7.5);	hold on;	% truncated Arnoldi
% [~, ind] = sort(SIZE(:, 3));
p2 = plot(SIZE(ind, 3), 'x', 'MarkerSize', 7.5);				% pinv
% [~, ind] = sort(SIZE(:, 7));
p3 = plot(SIZE(ind, 7), 's', 'MarkerSize', 7.5);				% OMP
% [~, ind] = sort(SIZE(:, 8));
p4 = plot(SIZE(ind, 8), 'd', 'MarkerSize', 7.5);				% SP
% [~, ind] = sort(SIZE(:, 9));
p5 = plot(SIZE(ind, 9), '^', 'MarkerSize', 7.5);				% Greedy
legend(leg2([1, 3, 7, 8, 9]),'Location','southeast','NumColumns',1)
% title(sprintf('%d matrices, m = %d, k = %d, cond < 10\\^%d, b = e\\_1', length(ids), p, t, floor(log10(condbound))));
if perturb
	title(sprintf('$%d$ matrices, $m = %d$, $k = %d$, $\\textrm{cond} < 10^{%d}$, $\\textit{\\textbf{b}} = \\textit{\\textbf{e}}_1 + 10^{-15} \\textit{\\textbf{e}}$', length(ids), p+1, t, floor(log10(condbound))));
else
	title(sprintf('$%d$ matrices, $m = %d$, $k = %d$, $\\textrm{cond} < 10^{%d}$, $\\textit{\\textbf{b}} = \\textit{\\textbf{e}}_1$', length(ids), p+1, t, floor(log10(condbound))));
end
xlabel('test matrix'), ylabel('basis size'); 
grid on;
grid minor;
if perturb
	fname = sprintf('ssa_plots/test2bdim_%d_m%d_s%d_k%d_cnd%d_e1pert_size',length(ids),p+1,s,t,floor(log10(condbound)));
else
	fname = sprintf('ssa_plots/test2bdim_%d_m%d_s%d_k%d_cnd%d_e1_size',length(ids),p+1,s,t,floor(log10(condbound)));
end
	% mypdf(fname,0.66,1.8)
	ylim([-5, p+6]);
mypdf(fname,0.66,0.85)
