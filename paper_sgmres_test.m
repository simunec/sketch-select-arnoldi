% Performance plots for sGMRES using different ways to construct the Krylov basis

% maximum cond(V) = 1e15, iterate until the basis condition number goes above the threshold, and record the norm of the residual that is reached

% Performance indicator: log(residual_norm), renormalized (see below for details)

% uses only full Arnoldi, truncated, pinv, OMP, SP, greedy

clear all
close all
clc

mydefaults
addpath('ssget')
index = ssget;
ids = find(index.numerical_symmetry==0 & index.ncols==index.nrows & index.ncols>=1e4 & index.ncols<1e6);
nids = length(ids);

load("ids_selected.mat");

testidx_actual = 0;
for testidx = 1:length(ids_selected)

    idj = ids_selected(testidx);
	% Uncomment to skip idj = 40 and idj = 55 (slowest test problems)
	% if (idj == 40 || idj == 55)
	% 	continue;
	% end
    close all

	% Only increment if not skipped:
	testidx_actual = testidx_actual + 1;

    Prob = ssget(ids(idj));
    fprintf('\nPROBLEM %s (%d of %d)\n', Prob.name, idj, length(ids));

    A = Prob.A;  
    n = size(A,1);

    A = @(x) A*x;

    rng('default');
    b = randn(n,1);
    
    b = b/norm(b);
    x0 = zeros(n,1);
	mmax = 600;
    m = mmax;
    every = 5;		
	maxcond = 1e15;
	restol = 1e-8;
	t = 5;
    s = 2*(m+1);
    rng('default')
    hS = srhtb2(n, s);


    %% standard Arnoldi
    fprintf('\nstandard Arnoldi\t|')
    V1 = b;	AV1 = [];
    H1 = zeros(m+1,m);
	reshist1 = [];
	for j = 1:m
		if ~mod(j, every)
			fprintf('.')
		end
        w = A(V1(:,j));
		AV1(:, j) = w;
        for i = 1:j
            for reo = 0:0
                h = V1(:,i)'*w;
                w = w - h*V1(:,i);
                H1(i,j) = H1(i,j) + h;
            end
        end
        H1(j+1,j) = norm(w);
        V1(:,j+1) = w/H1(j+1,j);
    
		if ~mod(j, every)
			% Check if converged:
			coeffs = H1(1:j+1,1:j)\eye(j+1,1);
			x = V1(:,1:j)*coeffs;
			reshist1(end+1) = norm(b - A(x));
			newres = norm(b - A(x));
			if (newres < restol)
				m = j;	% stop at j also for other methods
				fprintf('|')
				break;
			end

			% Monitor condition number: 
			cnd1 = cond(V1(:, 1:j));
			if (cnd1 >= maxcond)
				break;
			end
		end
	end
	% Compute final residual:
	its1(testidx_actual) = j;
	coeffs = H1(1:j+1,1:j)\eye(j+1,1);
	x = V1(:,1:j)*coeffs;
	res1(testidx_actual) = norm(b - A(x));


	%% truncated Arnoldi
    fprintf('\ntruncated Arnoldi\t|')
    V2 = b;	AV2 = [];
    H2 = zeros(m+1,m);
    SV2 = hS(V2); SAV2 = []; Sb = hS(b);
	reshist2 = [];
	for j = 1:m
		if ~mod(j, every)
			fprintf('.')
		end
        w = A(V2(:,j));
		AV2(:, j) = w;
        SAV2(:,j) = hS(w);
        for i = max(1,j-t+1):j
            H2(i,j) = V2(:,i)'*w;
            w = w - H2(i,j)*V2(:,i);
        end
        H2(j+1,j) = norm(w);
        V2(:,j+1) = w/H2(j+1,j);
        SV2(:,j+1) = hS(V2(:,j+1));
    
		if ~mod(j, every)
			% Compute residual history:
			[U, T] = qr(SAV2(:, 1:j), 0);
			x = V2(:,1:j) * (T \ (U'*Sb));	
			reshist2(end+1) = norm(b - A(x));
			
			% Monitor condition number: 
			cnd2 = cond(V2(:, 1:j));
			if (cnd2 >= maxcond)
				break;
			end
		end
	end
	% Compute final residual:
	its2(testidx_actual) = j;
	[U, T] = qr(SAV2(:, 1:j), 0);
	x = V2(:,1:j) * (T \ (U'*Sb));
	res2(testidx_actual) = norm(b - A(x));


	%% sketch-and-select Arnoldi pinv
    fprintf('\nssa with pinv\t\t|')
    V3 = b; AV3 = [];
    H3 = zeros(m+1,m);
    % use same t, s, hS
    SV3 = hS(V3); SAV3 = []; Sb = hS(b);
	reshist3 = [];
	for j = 1:m
		if ~mod(j, every)
			fprintf('.')
		end
        w = A(V3(:,j));
		AV3(:, j) = w;
        sw = hS(w); SAV3(:,j) = sw;
        % coeffs = pinv(SV3(:,1:j))*sw;
		[qq, rr] = qr(SV3(:, 1:j), 0);
		coeffs = rr \ (qq'*sw);
        [~,ind] = maxk(abs(coeffs),t);
        w = w - V3(:,ind)*coeffs(ind);
        H3(ind,j) = coeffs(ind);
        sw = hS(w);
        H3(j+1,j) = norm(sw);
        V3(:,j+1) = w/H3(j+1,j);
        SV3(:,j+1) = sw/H3(j+1,j);

		if ~mod(j, every)
			% Compute residual history:
			[U, T] = qr(SAV3(:, 1:j), 0);
			x = V3(:,1:j) * (T \ (U'*Sb));	
			reshist3(end+1) = norm(b - A(x));

			% Monitor condition number: 
			cnd3 = cond(V3(:, 1:j));
			if (cnd3 >= maxcond)
				break;
			end
		end
	end
	% Compute final residual:
	its3(testidx_actual) = j;
	[U, T] = qr(SAV3(:, 1:j), 0);
	x = V3(:,1:j) * (T \ (U'*Sb));
	res3(testidx_actual) = norm(b - A(x));


	%% ssa with OMP
    fprintf('\nssa with OMP\t\t|')
	reshist4 = [];
	V4 = b;	AV4 = [];
	H4 = zeros(m+1,m);
	% use same t, s, hS
	SV4 = hS(V4); SAV4 = []; Sb = hS(b);
	nsw = norm(SV4);
	SV4 = SV4/nsw;	V4 = V4/nsw;
	for j = 1:m
		if ~mod(j, every)
			fprintf('.')
		end
        w = A(V4(:,j));
		AV4(:, j) = w;
        sw = hS(w);
		SAV4(:,j) = sw;
		% OMP -- INITIALIZATION:
		r = sw;					% residual for OMP
		idx = zeros(0, 1);		% index set for orthogonalization
		SV4_i = zeros(s, 0);		% selected columns of SV
		x4_i = zeros(0, 1);		% to prevent errors for t = 0
		% OMP -- LOOP:
		for i = 1:min(j, t)		
            corr = abs(SV4(:,1:j)'*r);			% correlations, original version
			% corr = abs(pinv(SV4(:,1:j)) * r);	% alternative with pinv?
			corr(idx) = 0;					% to avoid picking the same column twice (when using variant)
			[~, idx_i] = max(corr);	% find column in SV4 with largest correlation with r
			idx = [idx, idx_i];				% add selected index
			SV4_i = [SV4_i, SV4(:, idx_i)];	% add selected column
			% x_i = pinv(SV4_i) * sw;			% "sparse" approximation to Sw (using columns of SV4_i)
			[qq_i, rr_i] = qr(SV4_i, 0);
			x_i = rr_i \ (qq_i'*sw);
	
			r = sw - SV4_i*x_i;				% updated residual
		end		% END OMP LOOP
		% at this point we have the index set idx for orthogonalization
		% Orthogonalize and update basis (same as other methods):
        % h = pinv(SV4(:,idx)) * sw;	
		[qq, rr] = qr(SV4(:, idx), 0);
		h = rr \ (qq'*sw);	
        w = w - V4(:,idx)*h;
        sw = sw - SV4(:,idx)*h;
		H4(idx,j) = h;
		nsw = norm(sw);
		H4(j+1,j) = nsw;
        V4(:,j+1) = w/nsw;
        SV4(:,j+1) = sw/nsw;

		if ~mod(j, every)
			% Compute residual history:
			[U, T] = qr(SAV4(:, 1:j), 0);
			x = V4(:,1:j) * (T \ (U'*Sb));	
			reshist4(end+1) = norm(b - A(x));

			% Monitor condition number: 
			cnd4 = cond(V4(:, 1:j));
			if (cnd4 >= maxcond)
				break;
			end
		end
	end
	% Compute final residual:
	its4(testidx_actual) = j;
	[U, T] = qr(SAV4(:, 1:j), 0);
	x = V4(:,1:j) * (T \ (U'*Sb));
	res4(testidx_actual) = norm(b - A(x));
    


    %% sketch-and-select Arnoldi SP
    fprintf('\nssa with SP\t\t|')
	reshist7 = [];
    V7 = b; AV7 = [];
    H7 = zeros(m+1,m);
    % use same t, s, hS
    SV7 = hS(V7); SAV7 = []; Sb = hS(b);
	itsp = 1;
	for j = 1:m
		if ~mod(j, every)
			fprintf('.')
		end
		w = A(V7(:,j));
		AV7(:, j) = w;
		sw = hS(w);
		SAV7(:,j) = sw;
		H7(:,j) = 0;
		% SP -- INITIALIZATION:
		corr = abs(SV7(:,1:j)'*sw);			% correlations, original version
		% corr = abs(pinv(SV(:,1:j)) * sw);		% alternative with pinv?
		[~, idx_i] = maxk(corr, min(j, t));	% find t columns in SV with largest correlation with r
		SV7_i = SV7(:, idx_i);			% select columns
		x_i = pinv(SV7_i) * sw;				% corresponds to recomputing the coefficients using those columns
		Sr = sw - SV7_i * x_i;			% compute residual
		% SP -- LOOP:
		for isp = 1:itsp
			y = SV7' * Sr;						% correlations of basis with residual, original version
			% y = pinv(SV)*Sr;						% alternative with pinv?
			[~, idx2_i] = maxk(abs(y), t);		% find t indices with largest components
			idxU_i = union(idx_i, idx2_i);	% union of old and new index sets
			% xU = pinv(SV7(:, idxU_i)) * sw;
			[qqU, rrU] = qr(SV7(:, idxU_i), 0);
			xU = rrU \ (qqU'*sw);
			% here, idx_rel is the index relative to the coordinates of xU!
			[~, idx_rel] = maxk(abs(xU), t);	% t (relative) indices with largest components in the union
			idx_i = idxU_i(idx_rel);			% get indices according to all columns
			SV7_i = SV7(:, idx_i);			% new column set
			% x_i = pinv(SV7_i) * sw;			% new coefficients
			[qq, rr] = qr(SV7_i, 0);
			x_i = rr \ (qq'*sw);	
			Sr = sw - SV7_i * x_i;			% new residual
		end		% END SP LOOP
		% at this point we have the index set idx_i for orthogonalization
		% Orthogonalize and update basis (same as other methods):
		% h = pinv(SV7(:,idx_i)) * sw;	
		[qq, rr] = qr(SV7(:, idx_i), 0);
		h = rr \ (qq'*sw);
		H7(idx_i,j) = h;
		w = w - V7(:,idx_i)*h;
		sw = hS(w);		% explicit sketch
		% sw = sw - SV7(:,idx_i)*h;
		H7(j+1,j) = norm(sw);
		V7(:,j+1) = w/H7(j+1,j);
		SV7(:,j+1) = sw/H7(j+1,j);

		if ~mod(j, every)
			% Compute residual history:
			[U, T] = qr(SAV7(:, 1:j), 0);
			x = V7(:,1:j) * (T \ (U'*Sb));	
			reshist7(end+1) = norm(b - A(x));

			% Monitor condition number: 
			cnd7 = cond(V7(:, 1:j));
			if (cnd7 >= maxcond)
				break;
			end
		end
	end
	% Compute final residual:
	its7(testidx_actual) = j;
	[U, T] = qr(SAV7(:, 1:j), 0);
	x = V7(:,1:j) * (T \ (U'*Sb));
	res7(testidx_actual) = norm(b - A(x));


    %% sketch-and-select Arnoldi Greedy (Natarajan)
    fprintf('\nssa with greedy\t\t|')
	reshist10 = [];
    V10 = b; AV10 = [];
    H10 = zeros(m+1,m);
    % use same t, s, hS
    SV10 = hS(V10); SAV10 = []; Sb = hS(b);
	for j = 1:m
		if ~mod(j, every)
			fprintf('.')
		end
		w = A(V10(:,j));
		AV10(:, j) = w;
		sw = hS(w);
		SAV10(:,j) = sw;
		H10(:,j) = 0;

		% get indices via greedy
		% see NATARAJAN paper
		ind = [];
		SV_g = SV10; sw_g = sw;
		for it = 1:min(j, t)
			corr = SV_g'*sw_g;
			[~,i] = max(abs(corr));
			ind = [ ind; i];
			sw_g = sw_g - SV_g(:,i)*(SV_g(:,i)'*sw_g);
			SV_g = SV_g - SV_g(:,i)*(SV_g(:,i)'*SV_g);
			SV_g = SV_g./vecnorm(SV_g);
			SV_g(:,ind) = 0;
		end
		h = pinv(SV10(:,ind))*sw; % recompute
		[qq, rr] = qr(SV10(:, ind), 0);
		h = rr \ (qq'*sw);
        H10(ind,j) = h;
        w = w - V10(:,ind)*h;
        % sw = sw - SV10(:,ind)*h;
		sw = hS(w);		% explicit sketch
		H10(j+1,j) = norm(sw);
		V10(:,j+1) = w/H10(j+1,j);
		SV10(:,j+1) = sw/H10(j+1,j);

		if ~mod(j, every)
			% Compute residual history:
			[U, T] = qr(SAV10(:, 1:j), 0);
			x = V10(:,1:j) * (T \ (U'*Sb));	
			reshist10(end+1) = norm(b - A(x));

			% Monitor condition number: 
			cnd10 = cond(V10(:, 1:j));
			if (cnd10 >= maxcond)
				break;
			end
		end
	end
	% Compute final residual:
	its10(testidx_actual) = j;
	[U, T] = qr(SAV10(:, 1:j), 0);
	x = V10(:,1:j) * (T \ (U'*Sb));
	res10(testidx_actual) = norm(b - A(x));



	fprintf('\n');

	%% Plots for each problem:
	iters = every:every:m;
	figure;
	p1 = semilogy(iters(1:length(reshist1)), reshist1); hold on;
	p2 = semilogy(iters(1:length(reshist2)), reshist2, '*-', 'MarkerIndices', length(reshist2));
	p3 = semilogy(iters(1:length(reshist3)), reshist3,'o--', 'MarkerIndices', length(reshist3));
	p4 = semilogy(iters(1:length(reshist4)), reshist4,'+-.', 'MarkerIndices', length(reshist4));
	p7 = semilogy(iters(1:length(reshist7)), reshist7,'^:', 'MarkerIndices', length(reshist7));
	p10 = semilogy(iters(1:length(reshist10)), reshist10,'>:', 'MarkerIndices', length(reshist10));

	title(sprintf('residual mat%d-t%d-m%d',idj,t, m));
	legend('GMRES', 'trunc', 'pinv', 'OMP', 'SP', 'greedy', 'Location','southwest');
	% fname = sprintf('ssa_plots/v1_t%d-m%d-singlemat%d.png', t, mmax, idj);
	% print(gcf,'-dpng','-r200', fname);

end % next problem


%% Plots:
its_done = [its1(:), its2(:), its3(:), its4(:), its7(:), its10(:)];
residuals = [res1(:), res2(:), res3(:), res4(:), res7(:), res10(:)];
% each row is a different test problem, each column a different method
logres = log10(residuals);
% normalize logres so that the best solver has 1, and the others have > 1
% find best solver (minimum in logres):
bestres = min(logres, [], 2);
logres2 = logres - bestres + 1;
% logres2(i,j) = 2 means that the residual of the j-th method is 10 times the residual of the best method on problem i
% logres2(i,j) = k means that the residual of the j-th method is 10^(k-1) times the residual of the best method on problem i

figure;
[th_max, h] = perfprof0(logres2);

legend('GMRES', 'sGMRES-truncated', 'sGMRES-ssa-pinv', 'sGMRES-ssa-OMP', 'sGMRES-ssa-SP', 'sGMRES-ssa-greedy', 'Location','southeast');
% legend(leg2,'Location','southeast','NumColumns',1)
title(sprintf('$%d$ matrices, $\\textrm{restol} = 10^{%d}$, $k = %d$', length(ids_selected), floor(log10(restol)), t));
xlabel('$\theta$'), ylabel('residual performance'); 

fname = sprintf('ssa_plots/test_sgmres_t%d-m%d', t, m);
mypdf(fname,0.66,0.85)
