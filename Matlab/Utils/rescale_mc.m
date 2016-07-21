function [X, y_lims_init] = rescale_mc(X,mask_train)

y_lims_init = [min(min(X(mask_train == 1))), max(max(X(mask_train == 1)))];

mean_train = mean(mean(X(mask_train == 1)));
X = X - mean_train*mask_train;



end