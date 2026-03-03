addpath('spm12');
addpath('spm12/toolbox/DEM');

dem_psychology('healthy');
dem_psychology('depressed');
dem_psychology('manic');

disp('Press any key to close figures and exit...');
pause;
