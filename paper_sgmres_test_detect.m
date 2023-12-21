
% Finds test matrices on which gmres converges quickly enough

clear all
close all
clc

mydefaults
addpath('ssget')
index = ssget;
ids = find(index.numerical_symmetry==0 & index.ncols==index.nrows & index.ncols>=1e4 & index.ncols<1e6);
nids = length(ids);

required_accuracy = 1e-3;	% keep only matrices for which GMRES reaches the required accuracy
ids_selected = [];

for idj = 1:length(ids)
    
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
    m = 200;
    every = 10;
    
    %% standard Arnoldi
    fprintf('\nstandard Arnoldi\t|')
    V1 = b;
    H1 = zeros(m+1,m);
    res1 = []; cnd1 = [];
    for j = 1:m
		if ~mod(j, every)
			fprintf('.')
		end
        w = A(V1(:,j));
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
    
	% CHeck if required accuracy is reached:
	if (res1(end) <= required_accuracy)
		ids_selected(end+1) = idj;
end

	% Saves selected indices:
	save('ids_selected.mat', 'ids_selected');

end % next problem