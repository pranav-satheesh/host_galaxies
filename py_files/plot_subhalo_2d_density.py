import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist
from cosmo_sim_tools import brahma as il_brahma
from cosmo_sim_tools.arepo_tools import twoDplot as twod
from cosmo_sim_tools.arepo_tools.mdot_to_Lbol import get_conversion_factor_arepo


def _align_coordinates(coordinates, centre, align=True, sim_basePath=None, snapnum=None):
    """
    Apply alignment transformation to coordinates (same as galaxy2Dplots).
    
    Parameters:
    -----------
    coordinates : array
        Particle coordinates
    centre : array
        Center position
    align : bool
        Whether to align to angular momentum
    sim_basePath : str
        Path to simulation (needed for align=True)
    snapnum : int
        Snapshot number (needed for align=True)
    
    Returns:
    --------
    array : aligned coordinates
    """
    pos = coordinates - centre
    
    if not align:
        return pos
    
    # Calculate alignment using angular momentum (same as galaxy2Dplots)
    from cosmo_sim_tools.arepo_tools.global_props import get_particle_data
    
    Ldata = get_particle_data(sim_basePath, snapnum, '02345', 
                             ['Coordinates', 'Masses', 'Velocities'])
    pos_L = Ldata['Coordinates'] - centre
    r = np.linalg.norm(pos_L, axis=1)
    pos_L = pos_L[r < 5]
    masses = Ldata['Masses'][r < 5]
    vel = Ldata['Velocities'][r < 5]
    
    totL = np.array([np.sum(masses * (pos_L[:, 1] * vel[:, 2] - pos_L[:, 2] * vel[:, 1])),
                     np.sum(masses * (pos_L[:, 2] * vel[:, 0] - pos_L[:, 0] * vel[:, 2])),
                     np.sum(masses * (pos_L[:, 0] * vel[:, 1] - pos_L[:, 1] * vel[:, 0]))])
    zaxis = totL / np.linalg.norm(totL)
    
    ct = zaxis[2] / np.sqrt(zaxis[0]**2 + zaxis[1]**2 + zaxis[2]**2)
    st = np.sqrt(1 - ct**2)
    cp = zaxis[0] / np.sqrt(zaxis[0]**2 + zaxis[1]**2)
    sp = zaxis[1] / np.sqrt(zaxis[0]**2 + zaxis[1]**2)
    
    x_rot = pos[:, 0] * ct * cp + pos[:, 1] * sp * ct - st * pos[:, 2]
    y_rot = -pos[:, 0] * sp + pos[:, 1] * cp
    z_rot = pos[:, 0] * st * cp + pos[:, 1] * st * sp + pos[:, 2] * ct
    
    return np.column_stack([x_rot, y_rot, z_rot])


def _apply_view_projection(coords, view='xy'):
    """
    Apply view projection to aligned coordinates.
    
    Parameters:
    -----------
    coords : array (N, 3)
        Aligned coordinates
    view : str
        View projection ('xy', 'xz', 'yz', etc.)
    
    Returns:
    --------
    tuple : (axis1, axis2, axis3) projected coordinates
    """
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    view_map = {
        'xy': (x, y, z),
        'yx': (y, x, z),
        'xz': (x, z, y),
        'zx': (z, x, y),
        'yz': (y, z, x),
        'zy': (z, y, x),
    }
    
    return view_map.get(view, (x, y, z))

def plot_single_subhalo(sim_basePath, snapnum, subhalo_id, p_type, 
                        box_length=50, box_width=50, box_height=50, 
                        nbins=256, outdir=None, hsml_key=False, plot_bh_overlay=True,
                        radiative_efficiency=0.1, view='xy'):
    """
    Plot 2D gas density for a single subhalo.
    
    Parameters:
    -----------
    sim_basePath : str
        Path to simulation data
    snapnum : int
        Snapshot number
    subhalo_id : int
        Subhalo ID to plot
    p_type : int or str
        Particle type (0=gas, 5=black holes, etc.)
    box_length, box_width, box_height : float
        Box dimensions in kpc
    nbins : int
        Number of bins for the 2D plot
    outdir : str
        Output directory (defaults to current directory)
    hsml_key : bool
        Whether to plot BH smoothing length circle
    plot_bh_overlay : bool
        Whether to overlay BH scatter plot sized by mass and colored by luminosity
    radiative_efficiency : float
        Radiative efficiency for mdot-to-luminosity conversion (default 0.1)
    view : str
        View projection ('xy', 'xz', 'yz', etc.)
    """
    p_type_idx = int(p_type)
    p_property = 'Density'
    
    if outdir is None:
        outdir = os.getcwd()
    os.makedirs(outdir, exist_ok=True)
    
    # Load subhalo data
    subhalo_data = il_brahma.groupcat.loadSubhalos_postprocessed(
        sim_basePath, snapnum, fields=['SubhaloPos', 'SubhaloLenType', 'SubhaloHalfmassRadType', 'SubhaloMassType']
    )
    subhalo_pos = subhalo_data['SubhaloPos']
    subhalo_len_type = subhalo_data['SubhaloLenType']
    subhalo_halfmass_rad_type = subhalo_data['SubhaloHalfmassRadType']
    subhalo_mass_type = subhalo_data['SubhaloMassType']
    
    centre = subhalo_pos[subhalo_id]
    counts_by_type = subhalo_len_type[subhalo_id]
    halfmass_by_type = subhalo_halfmass_rad_type[subhalo_id]
    mass_by_type = subhalo_mass_type[subhalo_id]
    
    print(f'Subhalo {subhalo_id} particle count (p_type={p_type_idx}): {int(counts_by_type[p_type_idx])}')
    print(f'Subhalo {subhalo_id} half-mass radius (p_type={p_type_idx}): {float(halfmass_by_type[p_type_idx]):.3f}')
    print(f'Subhalo {subhalo_id} mass (p_type={p_type_idx}): {float(mass_by_type[p_type_idx]):.3f}')
    
    if hsml_key:
        # Load BH data for smoothing length visualization
        bh_data = il_brahma.snapshot.loadSubset(sim_basePath, snapnum, 5, 
                                                fields=['Coordinates', 'BH_Hsml'])
        hsml = 0.0
        if len(bh_data['Coordinates']) > 0:
            central_bh_idx = np.argmin(cdist([centre], bh_data['Coordinates'])[0])
            hsml = bh_data['BH_Hsml'][central_bh_idx]
            print(f'Central BH smoothing length: {hsml:.3f}')
    
    # Load and process BH data for overlay
    bh_coords_all = None
    bh_masses_all = None
    bh_luminosities_all = None
    bh_coords_plot = None
    bh_masses_plot = None
    bh_lum_plot = None
    
    if plot_bh_overlay:
        # Load all BH data
        bh_full_data = il_brahma.snapshot.loadSubset(sim_basePath, snapnum, 5, 
                                                     fields=['Coordinates', 'BH_Mass', 'BH_Mdot'])
        if bh_full_data != {'count': 0} and len(bh_full_data['Coordinates']) > 0:
            bh_coords_all = bh_full_data['Coordinates']
            bh_masses_all = bh_full_data['BH_Mass']
            bh_mdot_all = bh_full_data['BH_Mdot']
            
            # Calculate luminosity from mdot
            lum_conversion = get_conversion_factor_arepo(radiative_efficiency)
            bh_luminosities_all = np.array(bh_mdot_all) * lum_conversion  # in erg/s
            
            # Filter BHs to those within ~2x halfmass radius of the subhalo
            # This is a reasonable criterion to select "subhalo BHs"
            halfmass_radius = halfmass_by_type[5]  # BH halfmass radius
            if halfmass_radius > 0:
                search_radius = halfmass_radius * 3  # Use 3x halfmass for selection
            else:
                search_radius = 50  # Fallback: use box size
            
            distances_from_center = np.linalg.norm(bh_coords_all - centre, axis=1)
            mask_bh_in_subhalo = distances_from_center < search_radius
            
            bh_coords_plot = bh_coords_all[mask_bh_in_subhalo]
            bh_masses_plot = bh_masses_all[mask_bh_in_subhalo]
            bh_lum_plot = bh_luminosities_all[mask_bh_in_subhalo]
            
            print(f'\nBH overlay info:')
            if len(bh_coords_plot) > 0:
                print(f'  Number of BHs in subhalo region: {len(bh_coords_plot)}')
                print(f'  BH mass range: {np.min(bh_masses_plot):.3e} - {np.max(bh_masses_plot):.3e} M_sun')
                print(f'  BH luminosity range: {np.min(bh_lum_plot):.3e} - {np.max(bh_lum_plot):.3e} erg/s')
            else:
                print(f'  No BHs found in subhalo region')
                plot_bh_overlay = False
        
    # Create 2D plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    gas_density = twod.galaxy2Dplots(
        path=sim_basePath,
        snapnum=snapnum,
        p_type=str(p_type_idx),
        particle_property=p_property,
        view=view,
        box_height=box_height,
        box_length=box_length,
        box_width=box_width,
        Nbins=nbins,
        method='binning',
        align=True,
        centre=centre,
        vmin=-4,
        vmax=4,
        showBH=False,
        fill=False,
        figure=fig,
        colorbar=True,
        axis=ax
    )
    
    cbar = fig.colorbar(gas_density, ax=ax)
    cbar.set_label(r'$\mathrm{Log(Density [M_{\odot}pc^{-2}])}$')
    
    # Add BH scatter overlay if requested
    if plot_bh_overlay and bh_coords_plot is not None and len(bh_coords_plot) > 0:
        # Transform BH coordinates with alignment
        bh_coords_aligned = _align_coordinates(
            bh_coords_plot, centre, align=True, 
            sim_basePath=sim_basePath, snapnum=snapnum
        )
        
        # Apply view projection
        bhaxis1, bhaxis2, bhaxis3 = _apply_view_projection(bh_coords_aligned, view=view)
        
        # Filter BHs to box bounds
        box_len_half = box_length / 2.0
        box_wid_half = box_width / 2.0
        box_hgt_half = box_height / 2.0
        
        mask_bh_in_box = (
            (bhaxis3 > -box_hgt_half) & (bhaxis3 < box_hgt_half) &
            (bhaxis1 > -box_len_half) & (bhaxis1 < box_len_half) &
            (bhaxis2 > -box_wid_half) & (bhaxis2 < box_wid_half)
        )
        
        if np.any(mask_bh_in_box):
            bhaxis1_plot = bhaxis1[mask_bh_in_box]
            bhaxis2_plot = bhaxis2[mask_bh_in_box]
            bh_masses_plot_in_box = bh_masses_plot[mask_bh_in_box]
            bh_lum_plot_in_box = bh_lum_plot[mask_bh_in_box]
            
            # Compute sizes: log scale normalized to ~1e7 solar masses
            # s = 50 * (log10(mass / 1e7) + 1) gives nice spread
            sizes = 50 * (np.log10(bh_masses_plot_in_box / 1e7) + 1)
            sizes = np.maximum(sizes, 10)  # Ensure minimum size of 10
            
            # Normalize luminosity to [0, 1] for colormap
            lum_min = np.min(bh_lum_plot_in_box)
            lum_max = np.max(bh_lum_plot_in_box)
            if lum_max > lum_min:
                lum_norm = (np.log10(bh_lum_plot_in_box) - np.log10(lum_min)) / (np.log10(lum_max) - np.log10(lum_min))
            else:
                lum_norm = np.ones_like(bh_lum_plot_in_box) * 0.5
            
            # Create scatter plot
            scatter = ax.scatter(bhaxis1_plot, bhaxis2_plot, s=sizes, c=lum_norm, 
                               cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5,
                               zorder=10)
            
            # Add colorbar for luminosity
            cbar_bh = fig.colorbar(scatter, ax=ax, pad=0.12)
            
            # Create custom tick labels for colorbar (show actual luminosity values)
            tick_positions = np.linspace(0, 1, 5)
            tick_labels = [f'{10**(np.log10(lum_min) + (np.log10(lum_max) - np.log10(lum_min)) * p):.1e}' 
                          for p in tick_positions]
            cbar_bh.set_ticks(tick_positions)
            cbar_bh.set_ticklabels(tick_labels, fontsize=8)
            cbar_bh.set_label(r'$\mathrm{Luminosity \ [erg/s]}$', fontsize=10)
            
            # Add annotation with BH statistics
            n_bh_box = np.sum(mask_bh_in_box)
            stats_text = f'BHs in box: {n_bh_box}\nM range: {np.min(bh_masses_plot_in_box):.2e}-{np.max(bh_masses_plot_in_box):.2e}'
            ax.text(0.03, 0.03, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add BH smoothing circle
    if hsml_key and hsml > 0:
        bh_circle = Circle((0.0, 0.0), hsml, fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(bh_circle)
        ax.text(0.03, 0.95, f'r= {hsml:.3f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
    
    ax.set_title(f'Subhalo {subhalo_id} at snapshot {snapnum}', fontsize=14)
    
    # Save figure
    outfile = os.path.join(outdir, f'subhalo_{subhalo_id}_snap{snapnum}_ptype{p_type_idx}.pdf')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {outfile}')

if __name__ == '__main__':
    if len(sys.argv) < 6:
        raise SystemExit(
            'Usage: python plot_subhalo_2d_density.py <sim_basePath> <snapnum> <subhalo_id> <p_type> <outdir> [hsml_key] [plot_bh_overlay] [radiative_efficiency] [view]'
        )
    
    sim_basePath = sys.argv[1]
    snapnum = int(sys.argv[2])
    subhalo_id = int(sys.argv[3])
    p_type = int(sys.argv[4])
    outdir = sys.argv[5]
    hsml_key = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else False
    plot_bh_overlay = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else True
    radiative_efficiency = float(sys.argv[8]) if len(sys.argv) > 8 else 0.1
    view = sys.argv[9] if len(sys.argv) > 9 else 'xy'
    
    
    plot_single_subhalo(sim_basePath, snapnum, subhalo_id, p_type, outdir=outdir, 
                       hsml_key=hsml_key, plot_bh_overlay=plot_bh_overlay,
                       radiative_efficiency=radiative_efficiency, view=view)