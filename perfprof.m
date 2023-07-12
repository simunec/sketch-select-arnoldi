function [th_max,h] = perfprof(A,th_max)
%PERFPROF  Performance profile.
%   [th_max, h] = PERFPROF(A,th_max) produces a
%   peformance profile for the data in the M-by-N matrix A,
%   where A(i,j) > 0 measures the performance of the j'th solver
%   on the i'th problem, with smaller values of A(i,j) denoting
%   "better".  For each solver theta is plotted against the
%   probability that the solver is within a factor theta of
%   the best solver over all problems, for theta on the interval
%   [1, th_max].
%   Set A(i,j) = NaN if solver j failed to solve problem i.
%   TH_MAX defaults to the smallest value of theta for which
%   all probabilities are 1 (modulo any NaN entries of A).
%   h is a vector of handles to the lines with h(j)
%   corresponding to the j'th solver.
%
% Copyright (c) 2016, Desmond J. Higham and Nicholas J. Higham.
% All rights reserved.

% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:

% * Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.

% * Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution.

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

minA = min(A,[],2);
if nargin < 2, th_max = max( max(A,[],2)./minA ); end
tol = sqrt(eps);  % Tolerance.

[m,n] = size(A);  % m problems, n solvers.
markers = {'-', '-.', '--', ':', '*-'}; % markers for the plot
l = length(markers);
for j = 1:n                  % Loop over solvers.

    col = A(:,j)./minA;      % Performance ratios.
    col = col(~isnan(col));  % Remove NaNs.
    if isempty(col), continue; end
    theta = unique(col)';    % Unique elements, in increasing order.
    r = length(theta);
    prob = sum( col(:,ones(r,1)) <= theta(ones(length(col),1),:) ) / m;
    % Assemble data points for stairstep plot.
    k = [1:r; 1:r]; k = k(:)';
    x = theta(k(2:end)); y = prob(k(1:end-1));

    % Ensure endpoints plotted correctly.
    if x(1) >= 1 + tol, x = [1 x(1) x]; y = [0 0 y]; end
    if x(end) < th_max - tol, x = [x th_max]; y = [y y(end)]; end
        h(j) = plot(x,y, markers{mod(j-1,l)+1}); hold on

end
hold off
xlim([1 th_max])
