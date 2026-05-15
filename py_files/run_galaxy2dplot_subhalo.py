import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist

from cosmo_sim_tools import brahma as il_brahma
from cosmo_sim_tools.arepo_tools import twoDplot as twod
from cosmo_sim_tools.arepo_tools import arepo_package as arepo


def get_pair_data(filename):
    """Read pair information from the exported HDF5 file.

    The file is expected to contain a ``samples`` group with one dataset per
    field. Rows are filtered to keep only pairs with both ``merger_Ngas`` and
    ``control_Ngas`` greater than 50, and duplicate entries are removed using
    ``(redshift, merger_snap, merger_subhalo_id, control_subhalo_id)``.
    """
    with h5py.File(filename, 'r') as handle:
        sample_group = handle['samples']
        redshift = np.asarray(sample_group['redshift'][:], dtype=float)
        merger_snap = np.asarray(sample_group['merger_snap'][:], dtype=np.int64)
        merger_subhalo_id = np.asarray(sample_group['merger_subhalo_id'][:], dtype=np.int64)
        control_subhalo_id = np.asarray(sample_group['control_subhalo_id'][:], dtype=np.int64)
        merger_ngas = np.asarray(sample_group['merger_Ngas'][:], dtype=np.int64)
        control_ngas = np.asarray(sample_group['control_Ngas'][:], dtype=np.int64)

    valid_mask = (merger_ngas > 100) & (control_ngas > 100)
    filtered_rows = np.column_stack((
        redshift[valid_mask],
        merger_snap[valid_mask],
        merger_subhalo_id[valid_mask],
        control_subhalo_id[valid_mask],
    ))

    unique_rows = np.unique(filtered_rows, axis=0)

    return (
        unique_rows[:, 0],
        unique_rows[:, 1].astype(np.int64),
        unique_rows[:, 2].astype(np.int64),
        unique_rows[:, 3].astype(np.int64),
    )

def main():
    if len(sys.argv) < 7:
        raise SystemExit(
            'Usage: python run_galaxy2dplot_subhalo.py '
            '<pair_hdf5_file> <sim_basePath> <box_length> <box_width> <box_height> <nbins> [outdir]'
        )

    plot_subhalo_list_filename = sys.argv[1]
    sim_basePath = sys.argv[2]
    box_length = float(sys.argv[3])
    box_width = float(sys.argv[4])
    box_height = float(sys.argv[5])
    nbins = int(sys.argv[6])
    outdir = sys.argv[7] if len(sys.argv) > 7 else os.path.dirname(os.path.abspath(plot_subhalo_list_filename))
    p_type = sys.argv[8] if len(sys.argv) > 8 else '0'
    p_type_idx = int(p_type)
    p_property = 'Density'

    box_length_label = str(int(box_length)) if float(box_length).is_integer() else str(box_length).replace('.', 'p')
    box_width_label = str(int(box_width)) if float(box_width).is_integer() else str(box_width).replace('.', 'p')
    box_height_label = str(int(box_height)) if float(box_height).is_integer() else str(box_height).replace('.', 'p')

    os.makedirs(outdir, exist_ok=True)

    redshifts, merger_snaps, merger_subhalo_ids, control_subhalo_ids = get_pair_data(plot_subhalo_list_filename)

    # Run only the first available pair as an example.
    for i in range(min(3, len(redshifts))):
        print(f'Processing redshift {redshifts[i]} with merger subhalo ID {merger_subhalo_ids[i]} and control subhalo ID {control_subhalo_ids[i]}')
        brahma_snap = merger_snaps[i]
        #merger host galaxy
        all_subhalo_pos = il_brahma.groupcat.loadSubhalos_postprocessed(sim_basePath, brahma_snap, fields=['SubhaloPos'])
        all_subhalo_len_type = il_brahma.groupcat.loadSubhalos_postprocessed(sim_basePath, brahma_snap, fields=['SubhaloLenType'])
        all_subhalo_halfmass_rad = il_brahma.groupcat.loadSubhalos_postprocessed(sim_basePath, brahma_snap, fields=['SubhaloHalfmassRad'])
        all_subhalo_halfmass_rad_type = il_brahma.groupcat.loadSubhalos_postprocessed(sim_basePath, brahma_snap, fields=['SubhaloHalfmassRadType'])
        merger_host_centre = all_subhalo_pos[merger_subhalo_ids[i]]
        control_host_centre = all_subhalo_pos[control_subhalo_ids[i]]

        merger_counts_by_type = all_subhalo_len_type[merger_subhalo_ids[i]]
        control_counts_by_type = all_subhalo_len_type[control_subhalo_ids[i]]

        merger_halfmass_total = float(all_subhalo_halfmass_rad[merger_subhalo_ids[i]])
        control_halfmass_total = float(all_subhalo_halfmass_rad[control_subhalo_ids[i]])
        merger_halfmass_by_type = all_subhalo_halfmass_rad_type[merger_subhalo_ids[i]]
        control_halfmass_by_type = all_subhalo_halfmass_rad_type[control_subhalo_ids[i]]

  
        print(
            f'Merger subhalo {merger_subhalo_ids[i]} particle count (p_type={p_type_idx}): '
            f'{int(merger_counts_by_type[p_type_idx])}'
        )
        print(
            f'Control subhalo {control_subhalo_ids[i]} particle count (p_type={p_type_idx}): '
            f'{int(control_counts_by_type[p_type_idx])}'
        )

        print(
            f'Merger subhalo {merger_subhalo_ids[i]} half-mass radius (total): {merger_halfmass_total:.3f}'
        )
        print(
            f'Control subhalo {control_subhalo_ids[i]} half-mass radius (total): {control_halfmass_total:.3f}'
        )
        print(
            f'Merger subhalo {merger_subhalo_ids[i]} half-mass radius (p_type={p_type_idx}): '
            f'{float(merger_halfmass_by_type[p_type_idx]):.3f}'
        )
        print(
            f'Control subhalo {control_subhalo_ids[i]} half-mass radius (p_type={p_type_idx}): '
            f'{float(control_halfmass_by_type[p_type_idx]):.3f}'
        )

        print(merger_host_centre, control_host_centre)
        
        merger_bh_data = il_brahma.snapshot.loadSubset(sim_basePath, brahma_snap, 5, fields=['Coordinates','BH_Hsml'])
        control_bh_data = il_brahma.snapshot.loadSubset(sim_basePath, brahma_snap, 5, fields=['Coordinates','BH_Hsml'])

        if len(merger_bh_data['Coordinates']) > 0:
            merger_host_central_bh_idx = np.argmin(cdist([merger_host_centre], merger_bh_data['Coordinates'])[0])
            print(f'Merger host central BH index: {merger_host_central_bh_idx}')
            merger_hsml = merger_bh_data['BH_Hsml'][merger_host_central_bh_idx]
        else:
            merger_hsml = 0.0

        if len(control_bh_data['Coordinates']) > 0:
            control_host_central_bh_idx = np.argmin(cdist([control_host_centre], control_bh_data['Coordinates'])[0])
            print(f'Control host central BH index: {control_host_central_bh_idx}')
            control_hsml = control_bh_data['BH_Hsml'][control_host_central_bh_idx]
        else:
            control_hsml = merger_hsml
        #control_host_central_bh_idx = np.argmin(cdist([control_host_centre], control_bh_data['Coordinates'])[0])
        
        fig, ax = plt.subplots(1,2,figsize=(12, 6))

        merger_gas_density = twod.galaxy2Dplots(
            path=sim_basePath,
            snapnum=brahma_snap,
            p_type=p_type,
            particle_property=p_property,
            view='xy',
            box_height=box_height,
            box_length=box_length,
            box_width=box_width,
            Nbins=nbins,
            method='binning',
            align=True,
            centre=merger_host_centre,
            vmin=-4,
            vmax=4,
            showBH=True,
            fill=False,
            figure=fig,
            colorbar=True,
            axis=ax[0]
        )
        control_gas_density = twod.galaxy2Dplots(
            path=sim_basePath,
            snapnum=brahma_snap,
            p_type=p_type,
            particle_property=p_property,
            view='xy',
            box_height=box_height,
            box_length=box_length,
            box_width=box_width,
            Nbins=nbins,
            method='binning',
            align=True,
            centre=control_host_centre,
            vmin=-4,
            vmax=4,
            showBH=True,
            fill=False,
            figure=fig,
            colorbar=True,
            axis=ax[1]
        )


        cbar1=fig.colorbar(merger_gas_density,ax=ax[0])
        cbar1.ax.tick_params(axis='y', direction='out')
        cbar1.set_label(r'$\mathrm{Log(Density [M_{\odot}pc^{-2}])}$')

        cbar2=fig.colorbar(control_gas_density,ax=ax[1])
        cbar2.ax.tick_params(axis='y', direction='out')
        cbar2.set_label(r'$\mathrm{Log(Density [M_{\odot}pc^{-2}])}$')


        if merger_hsml > 0:
            bh_circle_merger = Circle((0.0, 0.0), merger_hsml, fill=False, edgecolor='black', linewidth=1.5)
            ax[0].add_patch(bh_circle_merger)
        ax[0].text(
            0.03,
            0.95,
            f'r= {merger_hsml:.3f}',
            transform=ax[0].transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
        )

        if control_hsml > 0:
            bh_circle_control = Circle((0.0, 0.0), control_hsml, fill=False, edgecolor='black', linewidth=1.5)
            ax[1].add_patch(bh_circle_control)
        ax[1].text(
            0.03,
            0.95,
            f'r= {control_hsml:.3f}',
            transform=ax[1].transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
        )

        ax[0].set_title(f'Merger host subhalo:{merger_subhalo_ids[i]}')
        ax[1].set_title(f'Control host subhalo:{control_subhalo_ids[i]}')

        fig.suptitle(f'Gas density at z={redshifts[i]:.2f}', fontsize=16)

        outfile = os.path.join(
            outdir,
            f'galaxy2d_pair_{p_type}z{int(redshifts[i])}_MergerSub{int(merger_subhalo_ids[i])}_ControlSub{int(control_subhalo_ids[i])}_P{p_property}_L{box_length_label}_W{box_width_label}_H{box_height_label}.pdf'
        )

        fig.savefig(outfile)
        plt.close(fig)
        print(f'Saved: {outfile}')


if __name__ == '__main__':
    main()
