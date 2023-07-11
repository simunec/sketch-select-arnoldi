function S = srht(n, s)
	% S = srht(n, s)
	%* Generates a Subsampled Random Hadamard Transform, to be used as a random subspace embedding
	%
	% Input:
	% 	n is the original space size
	% 	s is the embedding dimension
	% Output:
	% 	S: function handle such that S(x) = SS*x, where SS is the s x n matrix corresponding to the randomized embedding
	%
	% This function is taken from Oleg Balabanov's randKrylov code (https://github.com/obalabanov/randKrylov)
	%
	% Copyright (c) 2022, Oleg Balabanov.

	% This program is free software: you can redistribute it and/or modify it under 
	% the terms of the GNU Lesser General Public License as published by the Free Software 
	% Foundation, either version 3 of the License, or (at your option) any later 
	% version.
	
	% This program is distributed in the hope that it will be useful, but WITHOUT ANY 
	% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
	% PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
	
	% You should have received a copy of the GNU Lesser General Public License along with 
	% this program. If not, see <https://www.gnu.org/licenses/>. 
	
    D = randi([0 1], n,1)*2 - 1;
    N = 2^ceil(log(n)/log(2));
    perm = randperm(N,s);
    select = @(t,ind) t(ind);
    S = @(t) (1/sqrt(s)) * select(myfwht(D.*t),perm);

	% Fast Walsh Hadamard Transform
	function z = myfwht(a)
		h = 1;
		n = length(a);
		N = 2^ceil(log(n)/log(2));
		z = zeros(N,1);
        
		z(1:n) = a;
		while h < N
			for i = 1:2*h:N
				for j = i:(i + h - 1)
					x = z(j);
					y = z(j + h);
					z(j) = x + y;
					z(j + h) = x - y;
				end
			end
			h = 2*h;
		end
	end

end
