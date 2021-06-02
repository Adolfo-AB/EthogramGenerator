import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


def main():
    # Create a Stamen terrain background instance.
    stamen_terrain = cimgt.Stamen('terrain')

    fig = plt.figure(figsize = (32,24))

    # Create a GeoAxes in the tile's projection.
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)

    #BBox = [-26.95371, -22.49597, 13.38982, 16.38232]

    BBox = [-24.7331, -24.5983, 14.9359, 15.0114]
    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent(BBox, crs=ccrs.Geodetic())

    # Add the Stamen data at zoom level 13.
    ax.add_image(stamen_terrain, 13)



    plt.show()


if __name__ == '__main__':
    main()