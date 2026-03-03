import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist

from cosmo_sim_tools import illustris as il
from cosmo_sim_tools.arepo_tools import twoDplot as twod


def _usage():
    print('Usage: python run_galaxy2dplot_subhalo.py <basepath> <snapnum> <subhalo_id> <outdir> [box_length box_width box_height [nbins vmin vmax ptype]]')


def _format_dim_for_name(value):
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace('.', 'p')


def main():
    # if len(sys.argv) not in {5, 8, 12}:
    #     _usage()
    #     sys.exit(1)

    basepath = sys.argv[1]
    snapnum = int(sys.argv[2])
    subhalo_id = int(sys.argv[3])
    outdir = sys.argv[4]
    box_length = float(sys.argv[5])
    box_width = float(sys.argv[6])
    box_height = float(sys.argv[7])

    nbins = int(sys.argv[8])
    vmin = float(sys.argv[9])
    vmax = float(sys.argv[10])
    p_property = str(sys.argv[11])

    method = 'binning'
    show_bh = True
    fill = False


    os.makedirs(outdir, exist_ok=True)

    subhalo_pos = il.groupcat.loadSubhalos(basepath, snapnum, fields=['SubhaloPos'])
    subhalo_halfmass_rad = il.groupcat.loadSubhalos(basepath, snapnum, fields=['SubhaloHalfmassRad'])
    centre = subhalo_pos[subhalo_id]
    halfmass_rad = float(subhalo_halfmass_rad[subhalo_id]) #ckpc/h
    bh_data = il.snapshot.loadSubset(basepath, snapnum, 5, fields=['Coordinates','BH_Hsml'])
    bh_coords = bh_data['Coordinates']
    distances = cdist([centre], bh_coords)[0]
    closest_bh_idx = np.argmin(distances)
    closest_bh_hsml = bh_data['BH_Hsml'][closest_bh_idx]


    # Create figure and axis explicitly to ensure proper coordination
    #fig, ax = plt.subplots(dpi=300)

    fig, ax = plt.subplots(1,2,figsize=(12, 6))

    gas_density = twod.galaxy2Dplots(
        path=basepath,
        snapnum=snapnum,
        p_type='0',
        particle_property=p_property,
        view='xy',
        box_height=box_height,
        box_length=box_length,
        box_width=box_width,
        Nbins=nbins,
        method=method,
        align=True,
        centre=centre,
        vmin=vmin,
        vmax=vmax,
        showBH=show_bh,
        fill=fill,
        figure=fig,
        colorbar=True,
        axis=ax[0]
    )

    star_density = twod.galaxy2Dplots(
    path=basepath,
    snapnum=snapnum,
    p_type='4',
    particle_property=p_property,
    view='xy',
    box_height=box_height,
    box_length=box_length,
    box_width=box_width,
    Nbins=nbins,
    method=method,
    align=True,
    centre=centre,
    vmin=vmin,
    vmax=vmax,
    showBH=show_bh,
    fill=fill,
    figure=fig,
    colorbar=True,
    axis=ax[0]
)


    # ax = plt.gca()
    cbar1=fig.colorbar(gas_density,ax=ax[0])
    cbar1.ax.tick_params(axis='y', direction='out')
    cbar1.set_label(r'$\mathrm{Log(Density [M_{\odot}pc^{-2}])}$')

    cbar2=fig.colorbar(star_density,ax=ax[1])
    cbar2.ax.tick_params(axis='y', direction='out')
    cbar2.set_label(r'$\mathrm{Log(Density [M_{\odot}pc^{-2}])}$')

    # if halfmass_rad > 0:
    #     halfmass_circle = Circle((0.0, 0.0), halfmass_rad, fill=False, edgecolor='white', linewidth=1.5)
    #     ax.add_patch(halfmass_circle)

    if closest_bh_hsml > 0:
        bh_circle_gas = Circle((0.0, 0.0), closest_bh_hsml, fill=False, edgecolor='black', linewidth=1.5)
        bh_circle_star = Circle((0.0, 0.0), closest_bh_hsml, fill=False, edgecolor='black', linewidth=1.5)
        ax[0].add_patch(bh_circle_gas)
        ax[1].add_patch(bh_circle_star)

    ax[0].set_title('Gas Density')
    ax[1].set_title('Stellar Density')
    # if halfmass_rad > 0:
    #     halfmass_circle = Circle((0.0, 0.0), halfmass_rad, fill=False, edgecolor='white', linewidth=1.5)
    #     ax.add_patch(halfmass_circle)

    box_length_label = _format_dim_for_name(box_length)
    box_width_label = _format_dim_for_name(box_width)
    box_height_label = _format_dim_for_name(box_height)

    fig.suptitle(f'Snapshot:{snapnum}, Subhalo:{subhalo_id}', fontsize=16)
    outfile = os.path.join(
        outdir,
        f'galaxy2d_subhalo_S{snapnum}_Sub{subhalo_id}_P{p_property}_L{box_length_label}_W{box_width_label}_H{box_height_label}.pdf'
    )

    fig.savefig(outfile)
    print(f'Saved: {outfile}')


if __name__ == '__main__':
    main()
