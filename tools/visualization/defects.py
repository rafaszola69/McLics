# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
defects.py - ParaView (pvbatch) script: 3D images of the defects of a director_field snapshot.

    pvbatch defects.py SNAPSHOT.csv LABEL NX NY NZ SHELL

NX, NY, NZ are the largest grid indices (number of sites minus one). The defects are drawn as the volume where the
local order parameter S is below 0.78 of its largest value (sites with pt >= 1 only), inside a translucent shell:
SHELL = free (the sites of the sample), sphere, hemisphere, slab (the box) or cylinder (axis along z). Four images
are written: iso_LABEL (perspective view), iso_x_LABEL, iso_y_LABEL and iso_z_LABEL (views along x, y and z).
Normally called by render_defects.sh. Works with ParaView 5.6 (Python 2) and 5.10 (Python 3).
"""
from paraview.simple import *
import math
import sys


def set_threshold(f, lo, hi):
    """Threshold range, for the old (ThresholdRange) and new (Lower/UpperThreshold) ParaView interfaces."""
    if hasattr(f, "LowerThreshold"):
        f.LowerThreshold = lo
        f.UpperThreshold = hi
        if hasattr(f, "ThresholdMethod"):
            f.ThresholdMethod = "Between"
    else:
        f.ThresholdRange = [lo, hi]


paraview.simple._DisableFirstRenderCameraReset()
print("%s -> iso_%s, geometry: %s" % (sys.argv[1], sys.argv[2], sys.argv[6]))
Nx = int(sys.argv[3])
Ny = int(sys.argv[4])
Nz = int(sys.argv[5])
geometry = sys.argv[6]
label = sys.argv[2]
reader = CSVReader(FileName=[sys.argv[1]])

renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
SetActiveView(renderView1)

grid = TableToStructuredGrid(Input=reader)
grid.WholeExtent = [0, Nx, 0, Ny, 0, Nz]
grid.XColumn = 'x'
grid.YColumn = 'y'
grid.ZColumn = 'z'
gridDisplay = Show(grid, renderView1)
renderView1.ResetCamera()

# sites of the sample (pt >= 1: bulk and all surfaces)
threshold1 = Threshold(Input=grid)
threshold1.Scalars = ['POINTS', 'pt']
set_threshold(threshold1, 1.0, 1.0e9)
threshold1.UpdatePipeline()
Smin, Smax = threshold1.PointData.GetArray("S").GetRange()
Sdefect = 0.78 * Smax
renderView1.Update()

# defects: volume with S below Sdefect
defects = IsoVolume(Input=threshold1)
defects.InputScalars = ['POINTS', 'S']
defects.ThresholdRange = [0, Sdefect]
defectsDisplay = Show(defects, renderView1)
sLUT = GetColorTransferFunction('S')
sLUT.RGBPoints = [Sdefect, 0.705882, 0.0156863, 0.14902, 0.7529410, Sdefect + 1e-7, 0.105882, 0.0156863, 0.74902]
sLUT.ScalarRangeInitialized = 1.0
defectsDisplay.Representation = 'Surface'
defectsDisplay.ColorArrayName = ['POINTS', 'S']
defectsDisplay.LookupTable = sLUT
defectsDisplay.SetScalarBarVisibility(renderView1, False)
renderView1.Update()
sPWF = GetOpacityTransferFunction('S')
sPWF.Points = [0.4000000059604645, 0.0, 0.5, 0.0, 0.45006102323532104, 1.0, 0.5, 0.0]
sPWF.ScalarRangeInitialized = 1
SetActiveSource(grid)

# translucent shell: an analytic surface scaled to the grid, or the sample itself (free)
if geometry == 'sphere':
    shell = Sphere()
    shell.Center = [0.5, 0.5, 0.5]
    shell.Radius = 0.5
    shell.ThetaResolution = 100
    shell.PhiResolution = 100
elif geometry == 'hemisphere':
    shell = Sphere()
    shell.Center = [0.5, 0.5, (Nz - 0.5 * Nz) / Nz]
    shell.Radius = 0.5
    shell.ThetaResolution = 100
    shell.PhiResolution = 100
    if Nz < Nx:
        shell.EndPhi = 2 * math.atan(2.0 * Nz / Nx) * 180 / math.pi
    else:
        shell.EndPhi = 2 * math.atan(2.0 * (Nz - 0.5 * Nx) / Nx) * 180 / math.pi + 90
elif geometry == 'slab':
    shell = Box()
    shell.XLength = 1.0
    shell.YLength = 1.0
    shell.ZLength = 1.0
    shell.Center = [0.5, 0.5, 0.5]
elif geometry == 'cylinder':
    shell = Cylinder()
    shell.Center = [0.5, 0.5, 0.5]
    shell.Radius = 0.5
    shell.Resolution = 100
elif geometry != 'free':
    sys.exit("SHELL must be free, sphere, hemisphere, slab or cylinder (got %s)" % geometry)

if geometry == 'free':
    shellDisplay = Show(threshold1, renderView1)
else:
    shellDisplay = Show(shell, renderView1)
shellDisplay.Representation = 'Surface'
shellDisplay.Scale = [1, 1, 1] if geometry == 'free' else [Nx, Ny, Nz]
if geometry == 'cylinder':
    shellDisplay.Orientation = [90.0, 0.0, 0.0]
    shellDisplay.Position = [-1, 0.0, 0.0]
if geometry == 'hemisphere':
    shellDisplay.Scale = [Nx, Ny, Nx]
renderView1.Update()
Hide(grid, renderView1)
shellDisplay.Opacity = 0.2
shellDisplay.DiffuseColor = [0.5, 1, 1]
renderView1.Background = [1, 1, 1]
if hasattr(renderView1, 'UseColorPaletteForBackground'):  # ParaView >= 5.10
    renderView1.UseColorPaletteForBackground = 0
renderView1.Update()

bounds = grid.GetDataInformation().GetBounds()
defectsDisplay.DiffuseColor = [0.2, 0.1, 1.0]
renderView1.CameraParallelProjection = 1

# perspective-like view from the (+x, +y) side
renderView1.CameraParallelScale = 0.55 * bounds[5]
renderView1.CameraPosition = [2.0 * bounds[1], 2.0 * bounds[3], bounds[5]]
renderView1.CameraFocalPoint = [-1.0 * bounds[1], -1.0 * bounds[3], 0 * bounds[5]]
renderView1.CameraViewUp = [0, 0, 1]
ResetCamera(renderView1)
SaveScreenshot('iso_' + label, renderView1, ImageResolution=[Ny * 20, Nz * 20])

# views along x, y and z
renderView1.CameraParallelScale = 0.55 * bounds[5]
renderView1.CameraPosition = [2.5 * bounds[1], 0.5 * bounds[3], 0.5 * bounds[5]]
renderView1.CameraFocalPoint = [1.0 * bounds[1], 0.5 * bounds[3], 0.5 * bounds[5]]
renderView1.CameraViewUp = [0, 0, 1]
SaveScreenshot('iso_x_' + label, renderView1, ImageResolution=[Ny * 20, Nz * 20])

renderView1.CameraParallelScale = 0.55 * bounds[5]
renderView1.CameraPosition = [0.5 * bounds[1], 2.5 * bounds[3], 0.5 * bounds[5]]
renderView1.CameraFocalPoint = [0.5 * bounds[1], 1 * bounds[3], 0.5 * bounds[5]]
renderView1.CameraViewUp = [0, 0, 1]
SaveScreenshot('iso_y_' + label, renderView1, ImageResolution=[Nx * 20, Nz * 20])

renderView1.CameraParallelScale = 0.55 * bounds[3]
renderView1.CameraPosition = [0.5 * bounds[1], 0.5 * bounds[3], 2.5 * bounds[5]]
renderView1.CameraFocalPoint = [0.5 * bounds[1], 0.5 * bounds[3], 1.0 * bounds[5]]
renderView1.CameraViewUp = [0, 1, 0]
SaveScreenshot('iso_z_' + label, renderView1, ImageResolution=[Nx * 20, Ny * 20])
