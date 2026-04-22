import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection
from decimal import Decimal, getcontext
from joblib import Parallel, delayed

# ==========================================
# 1) Parallel Worker Function
# ==========================================
def run_realization(realization_idx, num_cells_target, steps, L_str):
    """
    Executes a single realization of the Voronoi optimization with Periodic Boundary Conditions
    and weighted updates (W = L^2 / (P_i * P_j)).
    """
    start_time = time.time()
    
    # 1. Set precision locally inside the worker process
    getcontext().prec = 64
    
    # 2. Reconstruct exact Decimals
    L_dec = Decimal(L_str)
    L_float = float(L_dec)
    N = num_cells_target
    
    # 3. Create an independent random generator for this specific thread
    rng = np.random.default_rng(realization_idx)

    # --- Generate Random Points (Square PBC) ---
    # Core points inside the main box [-L/2, L/2] x [-L/2, L/2]
    pts_in = np.column_stack((rng.uniform(-L_float/2, L_float/2, N),
                              rng.uniform(-L_float/2, L_float/2, N)))
    
    pts_in_dec = np.array([[Decimal(str(x)), Decimal(str(y))] for x, y in pts_in], dtype=object)
    
    pts_dec = []
    shifts = []
    
    # Block 1: Add Core points (Indices 0 to N-1)
    for i in range(N):
        pts_dec.append(pts_in_dec[i])
        shifts.append(np.array([Decimal('0'), Decimal('0')], dtype=object))
        
    # Block 2: Add 8 Ghost boxes for PBC (Indices N to 9N-1)
    for dx in [-L_dec, Decimal('0'), L_dec]:
        for dy in [-L_dec, Decimal('0'), L_dec]:
            if dx == Decimal('0') and dy == Decimal('0'):
                continue
            for i in range(N):
                pts_dec.append(np.array([pts_in_dec[i][0] + dx, pts_in_dec[i][1] + dy], dtype=object))
                shifts.append(np.array([dx, dy], dtype=object))
                
    pts_dec = np.array(pts_dec, dtype=object)
    pts_float = np.array([[float(x), float(y)] for x, y in pts_dec])

    vor = Voronoi(pts_float)
    XYv_strained = np.array([[Decimal(str(v[0])), Decimal(str(v[1]))] for v in vor.vertices], dtype=object)

    # --- Filter and Format Cells ---
    # We define all 9N cells to maintain proper index tracking for edges
    cells = [None] * (9 * N)
    edge_to_cells = {}

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region and len(region) >= 3:
            cell_verts = region
            current_verts = XYv_strained[cell_verts]
            mean_x = sum(v[0] for v in current_verts) / Decimal(len(current_verts))
            mean_y = sum(v[1] for v in current_verts) / Decimal(len(current_verts))
            
            cells[i] = {
                'verts': cell_verts, 
                'geometric_centroid': np.array([mean_x, mean_y], dtype=object)
            }
            
            for u, v in zip(cell_verts, cell_verts[1:] + cell_verts[:1]):
                key = tuple(sorted((u, v)))
                edge_to_cells.setdefault(key, []).append(i)

    # --- Precalculate Weights: W = L^2 / (P_i * P_j) ---
    # 1. Calculate Perimeter for the N core cells (ghosts are strictly identical)
    perimeters = [Decimal('0')] * N
    for i in range(N):
        if cells[i] is None:
            continue
        peri = Decimal('0')
        c_verts = cells[i]['verts']
        for u, v in zip(c_verts, c_verts[1:] + c_verts[:1]):
            vu = XYv_strained[u]
            vv = XYv_strained[v]
            dx = vu[0] - vv[0]
            dy = vu[1] - vv[1]
            peri += (dx*dx + dy*dy).sqrt()
        perimeters[i] = peri

    # 2. Precalculate reflection weights and PBC shifts per neighbor pair (for core cells only)
    for i in range(N):
        reflection_info = []
        if cells[i] is None:
            continue
        c_verts = cells[i]['verts']
        for u, v in zip(c_verts, c_verts[1:] + c_verts[:1]):
            key = tuple(sorted((u, v)))
            cs = edge_to_cells.get(key, [])
            if len(cs) == 2:
                j = cs[0] if cs[1] == i else cs[1]
                
                vu = XYv_strained[u]
                vv = XYv_strained[v]
                dx = vu[0] - vv[0]
                dy = vu[1] - vv[1]
                L_sq = dx*dx + dy*dy
                
                core_j = j % N
                
                # W = L^2 / (P_i * P_j). Ghost cell perimeters match their core counterparts exactly.
                W = L_sq / (perimeters[i] * perimeters[core_j])
                
                shift_j = shifts[j]
                reflection_info.append((core_j, shift_j, u, v, W))
        
        cells[i]['reflection_info'] = reflection_info

    # --- Optimization / Reflective Math ---
    def reflect_point_over_line(p, a, b):
        ab0 = b[0] - a[0]
        ab1 = b[1] - a[1]
        denom = ab0*ab0 + ab1*ab1
        
        if denom < Decimal('1e-31'): 
            return p
        
        p_minus_a0 = p[0] - a[0]
        p_minus_a1 = p[1] - a[1]
        t = (p_minus_a0*ab0 + p_minus_a1*ab1) / denom
        
        r0 = Decimal('2.0') * (a[0] + t * ab0) - p[0]
        r1 = Decimal('2.0') * (a[1] + t * ab1) - p[1]
        return np.array([r0, r1], dtype=object)

    # We ONLY track and update the central N core seeds
    final_seeds = np.array([cells[i]['geometric_centroid'] for i in range(N)], dtype=object)
    
    log_means_run = np.zeros(steps)
    log_stds_run = np.zeros(steps)
    log_displacements = np.zeros(N, dtype=float)

    for step in range(steps):
        new_seeds = np.empty_like(final_seeds)
        displacements = np.zeros(N, dtype=float)
        
        for i in range(N):
            refinfo = cells[i].get('reflection_info', [])
            
            if refinfo:
                rx_sum = Decimal('0')
                ry_sum = Decimal('0')
                weight_sum = Decimal('0')
                
                # Apply precalculated weights and PBC shifts to reflections
                for core_j, shift_j, u, v, W in refinfo:
                    Np_pos = final_seeds[core_j] + shift_j
                    r = reflect_point_over_line(Np_pos, XYv_strained[u], XYv_strained[v])
                    rx_sum += r[0] * W
                    ry_sum += r[1] * W
                    weight_sum += W
                
                if weight_sum > Decimal('1e-31'):
                    new_seeds[i] = np.array([rx_sum / weight_sum, ry_sum / weight_sum], dtype=object)
                else:
                    new_seeds[i] = final_seeds[i]
            else:
                new_seeds[i] = final_seeds[i]

            diff0 = new_seeds[i][0] - final_seeds[i][0]
            diff1 = new_seeds[i][1] - final_seeds[i][1]
            dist = (diff0*diff0 + diff1*diff1).sqrt()
            displacements[i] = float(dist)
                
        # --- LOG DOMAIN STATISTICS FOR THIS RUN ---
        displacements_clipped = np.clip(displacements, 1e-32, None)
        log_displacements = np.log10(displacements_clipped)
        
        log_means_run[step] = np.mean(log_displacements)
        log_stds_run[step] = np.std(log_displacements)
        
        final_seeds = new_seeds

    # Extract final spatial X coordinates for 1D spatial analysis
    final_x = np.array([float(s[0]) for s in final_seeds])

    # --- Calculate Final Geometry (ONLY for realization 0 to save processing time) ---
    plot_data = None
    if realization_idx == 0:
        # Only plot the N central core cells
        poly_verts = [np.array([[float(v[0]), float(v[1])] for v in XYv_strained[cells[i]['verts']]]) for i in range(N)]
        plot_seeds = np.array([[float(s[0]), float(s[1])] for s in final_seeds])
        plot_data = (poly_verts, plot_seeds, log_displacements.copy(), N)

    end_time = time.time()
    exec_time = end_time - start_time

    return log_means_run, log_stds_run, final_x, log_displacements.copy(), plot_data, exec_time

# ==========================================
# 2) Main Execution Block
# ==========================================
if __name__ == '__main__':
    # Set precision for the main thread
    getcontext().prec = 64

    # Setup Parameters
    num_realizations = 1000         
    
    # Grab cell number from SLURM script argument, default to 10 if missing
    if len(sys.argv) > 1:
        num_cells_target = int(sys.argv[1])
    else:
        num_cells_target = 10  
        
    steps = 150 

    # Calculate required Square Box Length (L) based on target number of cells
    area_per_cell = Decimal(str(np.sqrt(3))) / Decimal('2.0')
    total_area = Decimal(num_cells_target) * area_per_cell
    L = total_area.sqrt()
    L_float = float(L)

    print(f"Starting parallel ensemble: {num_realizations} realizations, {num_cells_target} cells each.")
    print("Using all available CPU cores via joblib...")
    print("-" * 50)

    # --- Execute Parallel Jobs ---
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_realization)(i, num_cells_target, steps, str(L))
        for i in range(num_realizations)
    )

    # --- Aggregate Results ---
    ensemble_log_means = np.zeros((num_realizations, steps))
    ensemble_log_stds = np.zeros((num_realizations, steps))
    
    all_final_x = []
    all_final_log_disp = []
    all_exec_times = []
    final_plot_data = None

    for i, res in enumerate(results):
        ensemble_log_means[i, :] = res[0]
        ensemble_log_stds[i, :] = res[1]
        all_final_x.append(res[2])
        all_final_log_disp.append(res[3])
        if res[4] is not None:
            final_plot_data = res[4]
        all_exec_times.append(res[5])

    # Combine spatial data across all runs
    all_final_x = np.concatenate(all_final_x)
    all_final_log_disp = np.concatenate(all_final_log_disp)

    # Calculate the grand mean and run-to-run STD vs steps
    grand_mean_log = np.mean(ensemble_log_means, axis=0)
    grand_std_log = np.std(ensemble_log_means, axis=0)

    # Calculate spatial statistics vs X-Coordinate
    num_bins = 20
    bins = np.linspace(-L_float/2, L_float/2, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    spatial_mean_log = np.zeros(num_bins)
    spatial_std_log = np.zeros(num_bins)
    
    # Digitize distances into bins (returns indices 1 to num_bins)
    bin_indices = np.digitize(all_final_x, bins)
    
    for b in range(1, num_bins + 1):
        mask = (bin_indices == b)
        if np.any(mask):
            spatial_mean_log[b-1] = np.mean(all_final_log_disp[mask])
            spatial_std_log[b-1] = np.std(all_final_log_disp[mask])
        else:
            spatial_mean_log[b-1] = np.nan
            spatial_std_log[b-1] = np.nan

    # Calculate Average Execution Time
    avg_time_per_realization = np.mean(all_exec_times)

    # ==============================================
    # 3) Export Data to .npy
    # ==============================================
    # 1. Save execution time
    np.save('average_time.npy', np.array([avg_time_per_realization]))
    
    # 2. Save log RMS vs steps (Row 0: Means, Row 1: STDs)
    steps_data = np.vstack((grand_mean_log, grand_std_log))
    np.save('log_rms_vs_steps.npy', steps_data)
    
    # 3. Save log RMS vs space (Row 0: Bin Centers, Row 1: Means, Row 2: STDs)
    space_data = np.vstack((bin_centers, spatial_mean_log, spatial_std_log))
    np.save('log_rms_vs_space.npy', space_data)

    print("Ensemble generation complete.")
    print("Exported '.npy' files successfully.")
    print("-" * 50)

    # ==============================================
    # 4) Visualization (Generated from Saved Data)
    # ==============================================
    # --- Plot 1: Geometry (Using transient plot_data from realization 0) ---
    if final_plot_data is not None:
        poly_verts, plot_seeds, final_log_disp, Nc = final_plot_data

        plt.figure(figsize=(10, 10))
        pc = PolyCollection(poly_verts, array=final_log_disp, cmap='magma', edgecolor='black', lw=0.4)
        plt.gca().add_collection(pc)

        # Plot Square Boundary
        square_boundary = plt.Rectangle((-L_float/2, -L_float/2), L_float, L_float, 
                                        color='green', fill=False, linestyle='--', linewidth=2, label='Periodic Box')
        plt.gca().add_patch(square_boundary)

        plt.colorbar(pc, label=r'$\log_{10}$(Final Step Displacement)')
        plt.scatter(plot_seeds[:,0], plot_seeds[:,1], c='cyan', s=2, alpha=0.5, label='Optimized Seeds')

        plt.axis('equal')
        plt.title(f"Sample Geometry (Realization 0, {Nc} Weighted Periodic Cells)")
        plt.legend(loc='upper right')
        
        # Save instead of show for HPC environments
        plt.savefig('geometry_sample.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Load data for the remaining plots
    loaded_steps_data = np.load('log_rms_vs_steps.npy')
    plot_mean_steps = loaded_steps_data[0]
    plot_std_steps = loaded_steps_data[1]
    
    loaded_space_data = np.load('log_rms_vs_space.npy')
    plot_bin_centers = loaded_space_data[0]
    plot_spatial_mean = loaded_space_data[1]
    plot_spatial_std = loaded_space_data[2]

    # --- Plot 2: Ensemble Convergence vs Steps ---
    steps_arr = np.arange(steps)

    plt.figure(figsize=(8, 4))
    plt.errorbar(steps_arr, plot_mean_steps, yerr=plot_std_steps, fmt='-o', markersize=3, 
                 ecolor='lightgray', capsize=2, elinewidth=1, label=r'Grand Mean $\pm$ 1 Run-to-Run STD')

    plt.title(f"Weighted Ensemble Convergence Limit ({num_realizations} Realizations, {num_cells_target} Cells)")
    plt.xlabel("Optimization Step")
    plt.ylabel(r"$\log_{10}$(Displacement Magnitude)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('convergence_vs_steps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 3: Final Log RMS vs. X-Coordinate ---
    plt.figure(figsize=(8, 4))
    plt.errorbar(plot_bin_centers, plot_spatial_mean, yerr=plot_spatial_std, fmt='-s', markersize=4, 
                 color='purple', ecolor='plum', capsize=3, elinewidth=1.5, label=r'Mean $\pm$ 1 Pooled STD')

    plt.axvline(-L_float/2, color='green', linestyle='--', label='Box Boundary (-L/2)')
    plt.axvline(L_float/2, color='green', linestyle='--')
    plt.title(f"Spatial Distribution of Final Step Displacements ({num_cells_target} Cells)")
    plt.xlabel("X-Coordinate")
    plt.ylabel(r"$\log_{10}$(Final Displacement Magnitude)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('spatial_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Final stats directly from saved arrays
    final_grand_mean = plot_mean_steps[-1]
    final_grand_std = plot_std_steps[-1]
    loaded_avg_time = np.load('average_time.npy')[0]

    print(f"Number of Realizations: {num_realizations}")
    print(f"Number of Cells: {num_cells_target}")
    print(f"Average Execution Time per Realization: {loaded_avg_time:.4f} seconds")
    print(f"FINAL ENSEMBLE STATISTICS (at step {steps}):")
    print(f"Grand Mean of Log-Displacement: {final_grand_mean:.4f}")
    print(f"Run-to-Run STD of Log-Mean:     {final_grand_std:.4f} orders of magnitude")
    print(f"Equivalent Typical Displacement: {10**final_grand_mean:.2e}")
    print("Plots saved to current directory.")
