# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
slice.py - ParaView (pvbatch) script: image of the director field in one plane of the lattice.

    pvbatch slice.py PLANE.csv OUTPUT.png AXIS COLOR

PLANE.csv is one plane of a director_field snapshot (all sites with the same x, y or z; see extract_slices.py).
AXIS (x, y or z) is the normal of the plane. Each site with pt >= 1 is drawn as a cylinder along the director.
COLOR is S (green, and red where the local order parameter S < 0.438, i.e. at defects) or abs (colored by the
absolute value of the director component along AXIS: red in the plane, blue along the normal; useful for
cholesterics). Large planes are drawn with a subset of the sites (about 4000; 200000 with abs).
Normally called by render_slices.sh. Works with ParaView 5.6 (Python 2) and 5.10 (Python 3).
"""
from paraview.simple import *
import math
import sys

InputFile = sys.argv[1]
OutputFile = sys.argv[2]
axis = sys.argv[3]
colorMode = sys.argv[4]
k = "xyz".index(axis)
comp = "an" + axis          # |n_axis|
# in-plane directions (horizontal, vertical) of the image: bounds indices of their maxima
plane = {"x": (3, 5), "y": (1, 5), "z": (1, 3)}[axis]


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
reader = CSVReader(FileName=[InputFile])

points = TableToPoints(Input=reader)
points.XColumn = 'x'
points.YColumn = 'y'
points.ZColumn = 'z'

# director as a vector, and its absolute component along the normal of the plane
calculator1 = Calculator(Input=points)
calculator1.ResultArrayName = 'n'
calculator1.Function = 'nx*iHat+ny*jHat+nz*kHat'

calculator2 = Calculator(Input=calculator1)
calculator2.ResultArrayName = comp
calculator2.Function = 'abs(n%s)' % axis

Show(calculator1)
Bounds = calculator1.GetDataInformation().GetBounds()
oldNP = Bounds[plane[0]] * Bounds[plane[1]]
RescaledNP = 4000
Hide(calculator1)

# one cylinder per site; for large planes, a uniform subset of the sites
glyph1 = Glyph(Input=calculator2, GlyphType='Cylinder')
glyph1.OrientationArray = ['POINTS', 'n']
glyph1.ScaleArray = ['POINTS', 'No scale array']
glyph1.GlyphTransform = 'Transform2'
if oldNP > RescaledNP:
    NewSize = 1.4 * math.sqrt(oldNP / RescaledNP)
    try:
        glyph1.GlyphMode = 'Uniform Spatial Distribution'
    except ValueError:  # ParaView >= 5.9
        glyph1.GlyphMode = 'Uniform Spatial Distribution (Bounds Based)'
else:
    NewSize = 0.9
    glyph1.GlyphMode = 'All Points'
glyph1.ScaleFactor = NewSize
glyph1.MaximumNumberOfSamplePoints = RescaledNP
glyph1.GlyphType.Radius = 0.25
glyph1.GlyphTransform.Rotate = [0.0, 0.0, 90.0]

renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
SetActiveView(renderView1)
SetActiveSource(glyph1)

# only the sites of the sample (pt >= 1: bulk and all surfaces)
threshold1 = Threshold(Input=glyph1)
threshold1.Scalars = ['POINTS', 'pt']
set_threshold(threshold1, 1.0, 1.0e9)
threshold1.AllScalars = 1
threshold1.UseContinuousCellRange = 0
glyph1Display = Show(threshold1, renderView1)

glyph1Display.Representation = 'Surface'
glyph1Display.ColorArrayName = [None, '']
glyph1Display.DiffuseColor = [1.0, 0.6666666666666666, 1.0]
glyph1Display.Interpolation = 'Flat'
glyph1.GlyphType.Resolution = 20

renderView1.ResetCamera()
if colorMode == 'S':
    ColorBy(glyph1Display, ('POINTS', 'S'))
    glyph1Display.RescaleTransferFunctionToDataRange(True, False)
    sLUT = GetColorTransferFunction('S')
    sLUT.RGBPoints = [0.438, 0.705882, 0.0156863, 0.14902, 0.4381, 0.8, 1.5, 1.2]
    sLUT.ScalarRangeInitialized = 1.0
    glyph1Display.SetScalarBarVisibility(renderView1, False)
    renderView1.OrientationAxesVisibility = 0
elif colorMode == 'abs':
    ColorBy(glyph1Display, ('POINTS', comp))
    glyph1Display.RescaleTransferFunctionToDataRange(True, False)
    sLUT = GetColorTransferFunction(comp)
    sLUT.RGBPoints = [0., 0.7, 0.015, 0.14, 1, 0.14, 0.015, 0.7]
    glyph1Display.SetScalarBarVisibility(renderView1, False)
    renderView1.OrientationAxesVisibility = 0
    glyph1.MaximumNumberOfSamplePoints = 50 * RescaledNP
else:
    sys.exit("COLOR must be S or abs (got %s)" % colorMode)

# camera along AXIS, parallel projection, looking at the center of the plane
center = [Bounds[1] / 2, Bounds[3] / 2, Bounds[5] / 2]
position = list(center)
position[k] = 6 * Bounds[2 * k + 1]
renderView1.CameraPosition = position
renderView1.CameraFocalPoint = center
renderView1.CameraViewUp = [0.0, 1.0, 0.0] if axis == "z" else [0.0, 0.0, 1.0]
renderView1.CameraParallelScale = Bounds[plane[1]] / 1.95
renderView1.CameraParallelProjection = 1
renderView1.Background = [0.6, 0.6, 0.6]
if hasattr(renderView1, 'UseColorPaletteForBackground'):  # ParaView >= 5.10
    renderView1.UseColorPaletteForBackground = 0
imgScale = int(2000 / math.sqrt(Bounds[plane[0]] * Bounds[plane[1]]))
SaveScreenshot(OutputFile, renderView1, SaveAllViews=0,
               ImageResolution=[imgScale * int(Bounds[plane[0]]), imgScale * int(Bounds[plane[1]])],
               FontScaling='Scale fonts proportionally',
               SeparatorWidth=0,
               SeparatorColor=[0.0, 0.0, 0.0],
               OverrideColorPalette='',
               StereoMode='No change',
               TransparentBackground=0,
               ImageQuality=100)
