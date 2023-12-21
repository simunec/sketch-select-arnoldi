
% Tests convergence of sGMRES related to basis condition number and singular values
% Compares a lot of methods

clear all
close all
clc

mydefaults
addpath('ssget')
index = ssget;
ids = find(index.numerical_symmetry==0 & index.ncols==index.nrows & index.ncols>=1e4 & index.ncols<1e6);
nids = length(ids);

load("ids_selected.mat");

% for idj = 1:length(ids)
for idj = [46, 43]		
    
    close all

    Prob = ssget(ids(idj));
    fprintf('\nPROBLEM %s (%d of %d)\n', Prob.name, idj, length(ids));

    A = Prob.A;  
    n = size(A,1);

    A = @(x) A*x;

    rng('default');
    b = randn(n,1);
    
    b = b/norm(b);
    x0 = zeros(n,1);
    m = 300;
    every = 5;
	t = 5;
    s = 2*(m+1);
    rng('default')
    hS = srhtb2(n, s);


    %% standard Arnoldi
    fprintf('\nstandard Arnoldi\t|')
    V1 = b;	AV1 = [];
    H1 = zeros(m+1,m);
    res1 = []; cnd1 = [];
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
    
        if ~mod(j,every)
            coeffs = H1(1:j+1,1:j)\eye(j+1,1);
            x = V1(:,1:j)*coeffs;
            res1(end+1) = norm(b - A(x));
            cnd1(end+1) = 1;
        end
    end
	% Compute singular value distribution of final basis:
	m2 = floor(m/2);
	sing1 = svd(V1);
	singhalf1 = svd(V1(:, 1:m2));
    

    %% truncated Arnoldi
    fprintf('\ntruncated Arnoldi\t|')
    V2 = b;	AV2 = [];
    H2 = zeros(m+1,m);
    SV2 = hS(V2); SAV2 = []; Sb = hS(b);
    res2 = [];	cnd2 = [];
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

		% Compute solution with QR factorization instead of pinv:
		if ~mod(j,every)
			[U, T] = qr(SAV2(:, 1:j), 0);
			x = V2(:,1:j) * (T \ (U'*Sb));
			res2(end+1) = norm(b - A(x));
			cnd2(end+1) = cond(V2);
		end
    end
	% Compute singular value distribution of final basis:
	m2 = floor(m/2);
	sing2 = svd(V2);
	singhalf2 = svd(V2(:, 1:m2));

    %% sketch-and-select Arnoldi pinv
    fprintf('\nssa with pinv\t\t|')
    V3 = b; AV3 = [];
    H3 = zeros(m+1,m);
    % use same t, s, hS
    SV3 = hS(V3); SAV3 = []; Sb = hS(b);
    res3 = [];	cnd3 = [];
	for j = 1:m
		if ~mod(j, every)
			fprintf('.')
		end
        w = A(V3(:,j));
		AV3(:, j) = w;
        sw = hS(w); SAV3(:,j) = sw;
        % coeffs = pinv(SV3(:,1:j))*sw;
		[Q, R] = qr(SV3(:, 1:j), 0);
        coeffs = R \ (Q'*sw);
        [~,ind] = maxk(abs(coeffs),t);
        w = w - V3(:,ind)*coeffs(ind);
        H3(ind,j) = coeffs(ind);
        sw = hS(w);
        H3(j+1,j) = norm(sw);
        V3(:,j+1) = w/H3(j+1,j);
        SV3(:,j+1) = sw/H3(j+1,j);
    
		% Compute solution with QR factorization instead of pinv:
		if ~mod(j,every)
			[U, T] = qr(SAV3(:, 1:j), 0);
			x = V3(:,1:j) * (T \ (U'*Sb));
            res3(end+1) = norm(b - A(x));
            cnd3(end+1) = cond(V3);
		end
	end
	% Compute singular value distribution of final basis:
	m2 = floor(m/2);
	sing3 = svd(V3);
	singhalf3 = svd(V3(:, 1:m2));


	%% ssa with OMP
    fprintf('\nssa with OMP\t\t|')
	V4 = b;	AV4 = [];
	H4 = zeros(m+1,m);
	% use same t, s, hS
	SV4 = hS(V4); SAV4 = []; Sb = hS(b);
	nsw = norm(SV4);
	SV4 = SV4/nsw;	V4 = V4/nsw;
	res4 = [];	cnd4 = [];	
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
			x_i = pinv(SV4_i) * sw;			% "sparse" approximation to Sw (using columns of SV4_i)
			r = sw - SV4_i*x_i;				% updated residual
		end		% END OMP LOOP
		% at this point we have the index set idx for orthogonalization
		% Orthogonalize and update basis (same as other methods):
        h = pinv(SV4(:,idx)) * sw;		
        w = w - V4(:,idx)*h;
        sw = sw - SV4(:,idx)*h;
		H4(idx,j) = h;
		nsw = norm(sw);
		H4(j+1,j) = nsw;
        V4(:,j+1) = w/nsw;
        SV4(:,j+1) = sw/nsw;

		% Compute solution with QR factorization instead of pinv:
		if ~mod(j,every)
			[U, T] = qr(SAV4(:, 1:j), 0);
			x = V4(:,1:j) * (T \ (U'*Sb));
            res4(end+1) = norm(b - A(x));
            cnd4(end+1) = cond(V4);
		end
	end
	% Compute singular value distribution of final basis:
	m2 = floor(m/2);
	sing4 = svd(V4);
	singhalf4 = svd(V4(:, 1:m2));


    %% sketch-and-select Arnoldi SP
    fprintf('\nssa with SP\t\t|')
    V7 = b; AV7 = [];
    H7 = zeros(m+1,m);
    % use same t, s, hS
    SV7 = hS(V7); SAV7 = []; Sb = hS(b);
    res7 = [];	cnd7 = [];
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
			xU = pinv(SV7(:, idxU_i)) * sw;
			% here, idx_rel is the index relative to the coordinates of xU!
			[~, idx_rel] = maxk(abs(xU), t);	% t (relative) indices with largest components in the union
			idx_i = idxU_i(idx_rel);			% get indices according to all columns
			SV7_i = SV7(:, idx_i);			% new column set
			x_i = pinv(SV7_i) * sw;			% new coefficients
			Sr = sw - SV7_i * x_i;			% new residual
		end		% END SP LOOP
		% at this point we have the index set idx_i for orthogonalization
		% Orthogonalize and update basis (same as other methods):
		h = pinv(SV7(:,idx_i)) * sw;		
		H7(idx_i,j) = h;
		w = w - V7(:,idx_i)*h;
		sw = hS(w);		% explicit sketch
		% sw = sw - SV7(:,idx_i)*h;
		H7(j+1,j) = norm(sw);
		V7(:,j+1) = w/H7(j+1,j);
		SV7(:,j+1) = sw/H7(j+1,j);

		% Compute solution with QR factorization instead of pinv:
		if ~mod(j,every)
			[U, T] = qr(SAV7(:, 1:j), 0);
			x = V7(:,1:j) * (T \ (U'*Sb));
            res7(end+1) = norm(b - A(x));
            cnd7(end+1) = cond(V7);
		end
	end
	% Compute singular value distribution of final basis:
	m2 = floor(m/2);
	sing7 = svd(V7);
	singhalf7 = svd(V7(:, 1:m2));


	%% sketch-and-select with Greedy (Natarajan)
    fprintf('\nssa with greedy\t\t|')
	V8 = b;	AV8 = [];
	H8 = zeros(m+1,m);
	% use same t, s, hS
	SV8 = hS(V8); SAV8 = []; Sb = hS(b);
	nsw = norm(SV8);
	SV8 = SV8/nsw;	V8 = V8/nsw;
	res8 = [];	cnd8 = [];	
	for j = 1:m
		if ~mod(j, every)
			fprintf('.')
		end
        w = A(V8(:,j));
		AV8(:, j) = w;
        sw = hS(w);
		SAV8(:,j) = sw;

		ind = [];
		SV_g = SV8; sw_g = sw;
		for it = 1:min(j, t)
			corr = SV_g'*sw_g;
			[~,i] = max(abs(corr));
			ind = [ ind; i];
			sw_g = sw_g - SV_g(:,i)*(SV_g(:,i)'*sw_g);
			SV_g = SV_g - SV_g(:,i)*(SV_g(:,i)'*SV_g);
			SV_g = SV_g./vecnorm(SV_g);
			SV_g(:,ind) = 0;
		end
		% h = pinv(SV8(:,ind))*sw; % recompute
		[qq, rr] = qr(SV8(:, ind), 0);
		h = rr \ (qq'*sw);
        H8(ind,j) = h;
        w = w - V8(:,ind)*h;
        % sw = sw - SV8(:,ind)*h;
		sw = hS(w);		% explicit sketch
		H8(j+1,j) = norm(sw);
		V8(:,j+1) = w/H8(j+1,j);
		SV8(:,j+1) = sw/H8(j+1,j);

		% Compute solution with QR factorization instead of pinv:
		if ~mod(j,every)
			[U, T] = qr(SAV8(:, 1:j), 0);
			x = V8(:,1:j) * (T \ (U'*Sb));
            res8(end+1) = norm(b - A(x));
            cnd8(end+1) = cond(V8);
		end
	end
	% Compute singular value distribution of final basis:
	m2 = floor(m/2);
	sing8 = svd(V8);
	singhalf8 = svd(V8(:, 1:m2));


	%% Plots:
    fprintf('\n');
    iters = every:every:m;
    %%
	set(0, 'defaultfigureposition', [100 100 1200 1200*3/5]);
	figure;
	subplot(2, 2, 1)
	semilogy(iters, res1); hold on;
	semilogy(iters, res2);
	semilogy(iters, res3,'--');
	semilogy(iters, res4,'-.');
	semilogy(iters, res7,':');
	semilogy(iters, res8,':');
	xlabel("$m$");
	ylabel("residual");
	xlim([-5, 305]);
	ylim([1e-11, 1e1]);
	yticks([1e-10 1e-5 1]);

	subplot(2, 2, 2);
	semilogy(iters, cnd1); hold on;
	semilogy(iters, cnd2);
	semilogy(iters, cnd3,'--');
	semilogy(iters, cnd4,'-.');
	semilogy(iters, cnd7,':');
	semilogy(iters, cnd8,':');
	xlabel("$m$");
	ylabel("cond$(V_m)$");
	xlim([-5, 305]);
	ylim([1e-1 1e18]);
	yticks([1e0, 1e5, 1e10, 1e15])

	subplot(2, 2, 3);
	semilogy(sing1); hold on;
	semilogy(sing2);
	semilogy(sing3,'--');
	semilogy(sing4,'-.');
	semilogy(sing7,':');
	semilogy(sing8,':');
	xlabel("$j$");
	ylabel("$\sigma_j(V_{300})$");
	xlim([-5, 305]);
	ylim([1e-17, 1e1]);
	yticks([1e-15, 1e-10, 1e-5, 1]);

	subplot(2, 2, 4);
	semilogy(singhalf1); hold on;
	semilogy(singhalf2);
	semilogy(singhalf3,'--');
	semilogy(singhalf4,'-.');
	semilogy(singhalf7,':');
	semilogy(singhalf8,':');
	xlabel("$j$");
	ylabel("$\sigma_j(V_{150})$");
	xlim([-2.5, 152.5]);
	ylim([1e-17, 1e1]);
	yticks([1e-15, 1e-10, 1e-5, 1]);
	legend('GMRES', 'sGMRES-truncated', 'sGMRES-ssa-pinv', 'sGMRES-ssa-OMP', 'sGMRES-ssa-SP', 'sGMRES-ssa-greedy', 'Location','southwest');
	fname = sprintf('ssa_plots/paper_singval_mat%d-t%d',idj, t);
    mypdf(fname, 0.6, 1)
	
end % next problem