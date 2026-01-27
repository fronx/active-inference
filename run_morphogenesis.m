addpath('spm12');
addpath('spm12/toolbox/DEM');
DEM_morphogenesis

% Keep figures open for inspection
disp('Press any key to close figures and exit...');
pause;

% Create video from frames
if exist('morphogenesis_frames', 'dir')
    system('ffmpeg -y -framerate 8 -i morphogenesis_frames/frame_%03d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p morphogenesis.mp4');
    disp('Video saved to morphogenesis.mp4');
end
