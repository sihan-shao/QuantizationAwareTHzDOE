% Written by Sihan Shao, 20240314
% This is a script on how to use the CST-Matlab-API to generate a 3D
% hologram model based on x-y-z coordinates file

clc;
clear;

%First of all you need to include the main path where the m.files of the
%api are found in your computer
addpath(genpath('C:/Users/SHAO SIHAN/OneDrive - Aalto University/POsimulationCodes/HologramDesignCodes/CstApi'));


%This command is used to initiate the CST application, as you can see here, it is assigned to
%the cst variable. 
cst = actxserver('CSTStudio.Application');

%This command is used to open a new CST project. From now on this mws
%variable will be used to operate any matlab function that is used for the
%control of CST
mws = cst.invoke('NewMWS');


%Sets default units for your project. mm, GHz e.t.c you can open
%CSTDefaultUnits to see the exact selections or if you want I have
%CSTDefineUnits function in the api Home folder that you can use to assign you own units
CstDefaultUnits(mws) 


%Here I am calling Copper annealed lossy and FR4 lossy, after these two
%function I can just write their names as a string and assign them to any
%desired shape
CstCopperAnnealedLossy(mws)
CstFR4lossy(mws)

% Here we just generate a 3D model so we donot consider about the boundary
% condition and just focus on how to build the bircks

% Load the hologram (N x 3 point cloud)
Holo3D_coordinates = load('DOE_xyz_coordinates_20240815-131546.csv');

% Now the unit is m, we need to convert it to mm
X = Holo3D_coordinates(:,1) * 1e3;
Y = Holo3D_coordinates(:,2) * 1e3;
Z = Holo3D_coordinates(:,3) * 1e3;

[N2, ~] = size(Z);
N = sqrt(N2);
Height_map = reshape(Z, [N, N]);
dx = 1.0; % 1 mm
dy = dx;

W = dx * N; % Width of the Hologram
L = dy * N; % Length of the Hologram


%Cstbrick creates a brick for the hologram.
% Here we firstly creates a substrate with specific height
% We have to set the material but the final files we needed is the .stl for
% 3D printer. We can set any material and the material of the brick will be
%copper, that's because I already called CstCopperAnnealedLossy(mws) before

% defined the substrate with 2 mm
Dx = 94;
Dy = 94;
Hs = 2.0;   % Height of the substrate of Hologram (unit: mm)

Name = 'Substrate';
Componenet1 = 'component1';
material = 'Copper (annealed)';
Xrange = [-0.5*Dx 0.5*Dx];
Yrange = [-0.5*Dy 0.5*Dy];
Zrange = [-Hs 0];
Cstbrick(mws, Name, Componenet1, material, Xrange, Yrange, Zrange);

% define four holes to fix the hologram to support structure
Componenet2 = 'component2';
material = 'Copper (annealed)';

Num_holes = 4;
rh = 1.6;
xh = [-42.5, -42.5, 42.5, 42.5];
yh = [-42.5, 42.5, 42.5, -42.5];

Axis = 'Z';
OuterRadius = rh;
InnerRadius = 0;

for hole = 1: Num_holes
    Name = ['Hole' '_' num2str(hole)];
    Xcenter = xh(hole);
    Ycenter = yh(hole);
    Zrange = [-Hs 0];
    Cstcylinder(mws, Name, Componenet2, material, Axis, OuterRadius, InnerRadius, Xcenter, Ycenter, Zrange);
end



% Loop through cooridnates to create the unit based on the height map
Componenet3 = 'component3';
material = 'FR-4 (lossy)';
for x = 1: N
    for y = 1: N
    Name = ['Unit' '_' num2str(x) '_' num2str(y)];
    % Calculate the center of the current unit and start from the top-left
    % point which x-y cooridnate is (-0.5 * W, 0.5 * L)
    centerX = (-0.5 * W) + ((x-1) * dx) + (dx / 2);
    centerY = (0.5 * L) - ((y-1) * dy) - (dy / 2);
    % align with the x-y cooridnate in CST
    height = Height_map(y, x);
    
    Xrange = [centerX - dx/2, centerX + dx/2];
    Yrange = [centerY - dy/2, centerY + dy/2];
    Zrange = [0, height];
    Cstbrick(mws, Name, Componenet3, material, Xrange, Yrange, Zrange);

    end
end







