#!/usr/bin/env python
# coding: utf-8

# # Christmas Tree Convex Hulls

# In[1]:


# Setup

#NOTE THE LEDS ARE GRB COLOUR (NOT RGB)

# Specify whether using simulator and plotting, or using actual tree
USING_SIMULATOR = False

# Here are the libraries I am currently using:
import time
# Note: Not using these for the simulator
if not USING_SIMULATOR:
    import board
    import neopixel
import re
import math

if USING_SIMULATOR:
    # Libraries for plotting
    import matplotlib
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    get_ipython().run_line_magic('matplotlib', 'inline')

# ============ I broke the rules and imported one additional package ================
# ============ This makes it much easier to make and change colour schemes ==========
# To get colour palettes
from seaborn import color_palette

# You are welcome to add any of these:
# import random
# For coordinates
import numpy as np
# import scipy
# For convex hull
from scipy.spatial import ConvexHull
import sys

# If you want to have user changable values, they need to be entered from the command line
# so import sys sys and use sys.argv[0] etc
# some_value = int(sys.argv[0])

# ============================================================
# sys.argv[1] can provide the name of a seaborn colour palette
# https://seaborn.pydata.org/tutorial/color_palettes.html
# ============================================================


# In[2]:


# IMPORT THE COORDINATES (please don't break this bit)

coordfilename = "coords.txt"

fin = open(coordfilename,'r')
coords_raw = fin.readlines()

coords_bits = [i.split(",") for i in coords_raw]

coords = []

for slab in coords_bits:
    new_coord = []
    for i in slab:
        new_coord.append(int(re.sub(r'[^-\d]','', i)))
    coords.append(new_coord)

#set up the pixels (AKA 'LEDs')
PIXEL_COUNT = len(coords) # this should be 500

# Extract coordinates for plotting
coords_array = np.array(coords)

if not USING_SIMULATOR:
    pixels = neopixel.NeoPixel(board.D18, PIXEL_COUNT, auto_write=False)
else:
    # Our pretend xmas tree:
    xs = coords_array[:,0]
    ys = coords_array[:,1]
    zs = coords_array[:,2]
    
    # Colours must be in range [0,1] for the simulator.
    pixels = [ [0,0,0] for i in range(500) ]


# ## Code for nested convex hulls
# 
# The convex hull of a set of points is the smallest convex set that contains all the points (including those on the boundary.) For a finite set of points, the convex hull will be a convex polytope.
# 
# https://en.wikipedia.org/wiki/Convex_hull
# 
# This code finds the convex hull of Matt's Christmas tree. It then removes the points that are on the hull and finds the convex hull of the remaining points; this is repeated until all points have been processed. This creates sets of points that are nested within each other, like a matrioskha doll or an onion. There is probably a better algorithm in terms of time complexity for finding all the layers but this was pretty easy to code.
# 
# The python class below finds the nested hulls and allows easy assignment of colours to the different layers.
# 
# Related to the convex hull, the "alpha shape" is the same idea except the sets are allowed to be a "little bit" concave, as defined by a parameter alpha. This is useful in practical settings where we may want to be a little less strict about the layers and could give a more pleasing look.
# 
# Both the convex hull and alpha shape ideas are studied in the field of math/computer science known as "computational geometry" (and in other fields as well.)
# 
# https://en.wikipedia.org/wiki/Computational_geometry

# In[3]:


# Class to hold nested convex hulls
class NestedHulls:
    
    # Find nested convex hulls
    # Apparently there is a more efficient algorithm to find all nested hulls, but it's xmas
    def __init__(self, points, targets):
        self.targets = targets # This gives us a place to hang on to the pixels or whatever
                               # that can be assigned colours.
        self.hulls = [] # Ordered from outer to inner
        original_indices = np.arange(len(points)) # we have to manually keep track of the original indices
        
        while(len(points) >= 4): # In 3D, algorithm needs at least 4 points. (Think why?)
            ch = ConvexHull(points) # get the convex hull
            self.hulls.append(original_indices[ch.vertices]) # save the indices of the new hull
            original_indices = np.delete(original_indices, ch.vertices, axis=0) # remove the indices of the new hull
            points = np.delete(points, ch.vertices, axis=0) # remove the points of the new hull
            
        self.hulls.append(original_indices) # These are the "leftovers" in the middle
    
    # The "length" of a NestedHulls object is the number of nested hulls
    def __len__(self):
        return len(self.hulls)

    # Define the [] operator to easily assign colours (or whatever) to hulls
    def __setitem__(self, key, value):
        for i in self.hulls[key]:
            self.targets[i] = value


# In[4]:


# Compute nested convex hulls of Matt's xmas tree
hulls = NestedHulls(coords_array, pixels)

# Maximum brightness if using actual tree
max_brightness = 50

# palette name for seaborne
# https://seaborn.pydata.org/tutorial/color_palettes.html
default_palette = "dark:seagreen"
try:
    c_palette = color_palette(sys.argv[1], len(hulls))
except:
    # Basically if there's no sys.argv[1] or it's not a palette
    c_palette = color_palette(default_palette, len(hulls))

if USING_SIMULATOR:
    # RGb range from 0 to 1 if using simulator
    colours = c_palette
    # colours works like a function that takes an input and produces a colour
else:
    # GRb from 0 to max_brightness if using tree
    colours = lambda i: (np.array(c_palette[i]) * max_brightness)[[1,0,2]].astype(int)


# In[5]:


# Display the colour palette on the tree, shifting "inward" by one hull every step

# pause between cycles (normally zero as it is already quite slow)
slow = 0

offset = 0 # Shift colours by this many hulls

# This try just suppresses all the junk associated with quitting with ctrl-c or "stop"
try:
    while True:

        # Pause for a bit
        time.sleep(slow)

        # Assign colours to each hull
        for h in range(len(hulls)):
            hulls[h] = colours[(h + offset) % len(hulls)]

        # Shift offset
        offset = (offset + 1) % len(hulls)

        # Display
        if not USING_SIMULATOR:
            pixels.show()
        else:
            # plot our fake xmas tree instead - plot appears below and is animated
            clear_output(wait=True)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs, color=pixels)
            plt.show()
        
except KeyboardInterrupt:
    print("Done!")


# In[ ]:




