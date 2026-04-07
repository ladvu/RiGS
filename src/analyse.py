import torch 
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import fsolve
import trimesh


def fit_gaussian_mixture(lifespan_data, n_components=2, random_state=42):
    data = lifespan_data.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(data)
    
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_
    
    sorted_indices = np.argsort(means)
    means = means[sorted_indices]
    stds = stds[sorted_indices]
    weights = weights[sorted_indices]
    
    return gmm, means, stds, weights


def find_local_minimum_between_means(means, stds, weights, num=2000):
    # assume means are sorted ascending
    mu1, mu2 = means[0], means[1]
    if mu1 == mu2:
        return None, None
    left = mu1
    right = mu2
    x = np.linspace(left, right, num)
    pdf_sum = weights[0] * norm.pdf(x, means[0], stds[0]) + weights[1] * norm.pdf(x, means[1], stds[1])
    idx_min = np.argmin(pdf_sum)
    return x[idx_min], pdf_sum[idx_min]

def visualize_gmm_lifespan(lifespan_data, scene_name="", bins=50, save_path=None, percentile_cutoff=95):
    # Remove long tail by using percentile cutoff
    cutoff_value = np.percentile(lifespan_data, percentile_cutoff)
    filtered_data = lifespan_data[lifespan_data <= cutoff_value]
    
    print(f"Original data points: {len(lifespan_data)}")
    print(f"After removing long tail (>{percentile_cutoff}th percentile): {len(filtered_data)}")
    print(f"Cutoff value: {cutoff_value:.4f}")
    
    gmm, means, stds, weights = fit_gaussian_mixture(filtered_data)
    
    # find local minimum (valley) between the two component means and use it as threshold
    valley_x, valley_y = find_local_minimum_between_means(means, stds, weights)
    if valley_x is not None:
        print(f"Local minimum between means: x={valley_x:.4f}, pdf={valley_y:.6f}")
    else:
        print("Could not compute local minimum (means identical?)")
    
    plt.figure(figsize=(12, 8))
    
    counts, bin_edges, patches = plt.hist(filtered_data, bins=bins, density=True, 
                                         alpha=0.6, color='lightblue', edgecolor='black', 
                                         label='Temporal Duration Distribution')
    
    x_range = np.linspace(filtered_data.min(), filtered_data.max(), 1000)
    
    gmm_pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
    plt.plot(x_range, gmm_pdf, 'r-', linewidth=3, label='GMM Fit', alpha=0.8)
    
    component1_pdf = weights[0] * norm.pdf(x_range, means[0], stds[0])
    component2_pdf = weights[1] * norm.pdf(x_range, means[1], stds[1])
    
    # plt.plot(x_range, component1_pdf, '--', linewidth=2, color='green', 
    #          label=f'Component 1: μ={means[0]:.3f}, σ={stds[0]:.3f}')
    # plt.plot(x_range, component2_pdf, '--', linewidth=2, color='orange', 
    #          label=f'Component 2: μ={means[1]:.3f}, σ={stds[1]:.3f}')
    
    # plot valley (local minimum) as the threshold
    # if valley_x is not None:
    #     plt.plot(valley_x, valley_y, 'ks', markersize=8, label=f'Local minimum threshold: x={valley_x:.3f}')
    #     plt.axvline(x=valley_x, color='black', linestyle='--', alpha=0.8)
    #     plt.annotate(f'x={valley_x:.3f}', xy=(valley_x, valley_y), xytext=(10, -30),
    #                  textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
    #                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # plt.xlabel('Temporal Duration', fontsize=20, fontweight='bold')
    # plt.ylabel('Probability Density', fontsize=20, fontweight='bold')
    # plt.legend(fontsize=16)
    # plt.grid(True, alpha=0.3)
    # plt.tick_params(axis='both', which='major', labelsize=16, width=2)
    # plt.gca().tick_params(axis='both', which='major', labelsize=16, colors='black')
    # Make tick labels bold
    # for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    #     label.set_fontweight('bold')
    
    # Remove axes, ticks, and frame
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
    
    return gmm, means, stds, weights, (valley_x, valley_y)

if __name__ == "__main__":
    # nvidia_result_dir = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/test_ground/nvidia_vis/TPGS/results_nvidia_vis"
    # scene_names = os.listdir(nvidia_result_dir)
    # for scene_name in scene_names:
    #     try:
    #         ckpt_dir = os.path.join(nvidia_result_dir, scene_name, "ckpts")
    #         ckpts = os.listdir(ckpt_dir)
    #         steps = [int(ckpt.split('_')[1].split('.')[0]) for ckpt in ckpts]
    #         sorted_indices = np.argsort(steps)
    #         ckpts = [ckpts[i] for i in sorted_indices]
    #         for i, ckpt in enumerate(ckpts):
    #             ckpt_path = os.path.join(ckpt_dir, ckpt)
    #             checkpoint = torch.load(ckpt_path, map_location='cpu')
    #             lifespan_data = checkpoint["dynamic_splats"]["lifespan"].cpu().numpy()
    #             save_dir = os.path.join("./misc/", f"nvidia_{scene_name}")
    #             os.makedirs(save_dir, exist_ok=True)
    #             _, _, _, _, xy = visualize_gmm_lifespan(lifespan_data, scene_name=scene_name,
    #                                    save_path=os.path.join(save_dir, f"{i:05d}.png"), 
    #                                    percentile_cutoff=95)
    #     except:
    #         continue
    nvidia_result_dir = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/test_ground/nvidia_wo_transient/TPGS/results_nvidia_wo_transient/"
    scene_names = os.listdir(nvidia_result_dir)
    for scene_name in scene_names:
        try:
            ckpt_path = os.path.join(nvidia_result_dir, scene_name, "ckpts", "ckpt_29999.pt")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            lifespan_data = checkpoint["dynamic_splats"]["lifespan"].cpu().numpy()
            _, _, _, _, xy = visualize_gmm_lifespan(lifespan_data, scene_name=scene_name,
                                   save_path=f"./misc/nvidia_{scene_name}_dist.pdf", 
                                   percentile_cutoff=95)
            x = xy[0]
            mask = checkpoint["dynamic_splats"]['lifespan'] < x
            pts = checkpoint["dynamic_splats"]["means"]
            clrs = torch.sigmoid(checkpoint["dynamic_splats"]["colors"])
            pts_t = pts[mask]
            clrs_t = clrs[mask]
            pts_r = pts[~mask]
            clrs_r = clrs[~mask]

            clrs_t_red = torch.zeros_like(clrs_t)
            clrs_t_red[:, 0] = 1.0
            clrs_r_blue = torch.zeros_like(clrs_r)
            clrs_r_blue[:, 2] = 1.0
            trimesh.PointCloud(vertices=pts_t, colors=clrs_t).export(f"./misc/nvidia_{scene_name}_transient.ply")
            trimesh.PointCloud(vertices=pts_r, colors=clrs_r).export(f"./misc/nvidia_{scene_name}_rigid.ply")

            trimesh.PointCloud(
                vertices=torch.cat([pts_t, pts_r], dim=0), 
                colors=torch.cat([clrs_t_red, clrs_r_blue], dim=0)
            ).export(f"./misc/nvidia_{scene_name}_transient_rigid_color_coded.ply")
            
        except:
            continue

    # iphone_result_dir = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/test_ground/iphone_longer_no_transient_new_new_new_2/TPGS/results_iphone_longer_no_transient_new_new_new_2/"
    # scene_names = os.listdir(iphone_result_dir)
    # for scene_name in scene_names:
    #     try:
    #         ckpt_path = os.path.join(iphone_result_dir, scene_name, "ckpts", "ckpt_99999.pt")
    #         checkpoint = torch.load(ckpt_path, map_location='cpu')
    #         lifespan_data = checkpoint["dynamic_splats"]["lifespan"].cpu().numpy()
    #         visualize_gmm_lifespan(lifespan_data, scene_name=scene_name,
    #                                save_path=f"./misc/iphone_{scene_name}_dist.pdf", 
    #                                percentile_cutoff=95)
    #         x = xy[0]
    #         mask = checkpoint["dynamic_splats"]['lifespan'] < x
    #         pts = checkpoint["dynamic_splats"]["means"]
    #         clrs = torch.sigmoid(checkpoint["dynamic_splats"]["colors"])
    #         pts_t = pts[mask]
    #         clrs_t = clrs[mask]
    #         pts_r = pts[~mask]
    #         clrs_r = clrs[~mask]

    #         clrs_t_red = torch.zeros_like(clrs_t)
    #         clrs_t_red[:, 0] = 1.0
    #         clrs_r_blue = torch.zeros_like(clrs_r)
    #         clrs_r_blue[:, 2] = 1.0
    #         trimesh.PointCloud(vertices=pts_t, colors=clrs_t).export(f"./misc/iphone_{scene_name}_transient.ply")
    #         trimesh.PointCloud(vertices=pts_r, colors=clrs_r).export(f"./misc/iphone_{scene_name}_rigid.ply")

    #         trimesh.PointCloud(
    #             vertices=torch.cat([pts_t, pts_r], dim=0), 
    #             colors=torch.cat([clrs_t_red, clrs_r_blue], dim=0)
    #         ).export(f"./misc/iphone_{scene_name}_transient_rigid_color_coded.ply")
    #     except:
    #         continue

    custom_result_dir = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/test_ground/custom_no_transient/TPGS/results_custom_no_transient"
    scene_names = os.listdir(custom_result_dir)
    for scene_name in scene_names:
        try:
            ckpt_path = os.path.join(custom_result_dir, scene_name, "ckpts", "ckpt_14.pt")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            lifespan_data = checkpoint["dynamic_splats"]["lifespan"].cpu().numpy()
            _, _, _, _, xy = visualize_gmm_lifespan(lifespan_data, scene_name=scene_name,
                                   save_path=f"./misc/custom_{scene_name}_dist.pdf", 
                                   percentile_cutoff=95)
            x = xy[0]
            mask = checkpoint["dynamic_splats"]['lifespan'] < x
            pts = checkpoint["dynamic_splats"]["means"]
            clrs = torch.sigmoid(checkpoint["dynamic_splats"]["colors"])
            pts_t = pts[mask]
            clrs_t = clrs[mask]
            pts_r = pts[~mask]
            clrs_r = clrs[~mask]

            clrs_t_red = torch.zeros_like(clrs_t)
            clrs_t_red[:, 0] = 1.0
            clrs_r_blue = torch.zeros_like(clrs_r)
            clrs_r_blue[:, 2] = 1.0
            trimesh.PointCloud(vertices=pts_t, colors=clrs_t).export(f"./misc/custom_{scene_name}_transient.ply")
            trimesh.PointCloud(vertices=pts_r, colors=clrs_r).export(f"./misc/custom_{scene_name}_rigid.ply")

            trimesh.PointCloud(
                vertices=torch.cat([pts_t, pts_r], dim=0), 
                colors=torch.cat([clrs_t_red, clrs_r_blue], dim=0)
            ).export(f"./misc/custom_{scene_name}_transient_rigid_color_coded.ply")
        except:
            continue

    custom_result_dir = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/test_ground/custom_transient_bce_new_demo_2/TPGS/results_custom_transient_bce_new_demo_2"
    scene_names = os.listdir(custom_result_dir)
    for scene_name in scene_names:
        try:
            ckpt_path = os.path.join(custom_result_dir, scene_name, "ckpts", "ckpt_11998.pt")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            lifespan_data = checkpoint["dynamic_splats"]["lifespan"].cpu().numpy()
            _, _, _, _, xy = visualize_gmm_lifespan(lifespan_data, scene_name=scene_name,
                                   save_path=f"./misc/custom_{scene_name}_dist.pdf", 
                                   percentile_cutoff=95)
            x = xy[0]
            mask = checkpoint["dynamic_splats"]['lifespan'] < x 
            pts = checkpoint["dynamic_splats"]["means"]
            clrs = torch.sigmoid(checkpoint["dynamic_splats"]["colors"])
            pts_t = pts[mask]
            clrs_t = clrs[mask]
            pts_r = pts[~mask]
            clrs_r = clrs[~mask]

            clrs_t_red = torch.zeros_like(clrs_t)
            clrs_t_red[:, 0] = 1.0
            clrs_r_blue = torch.zeros_like(clrs_r)
            clrs_r_blue[:, 2] = 1.0
            trimesh.PointCloud(vertices=pts_t, colors=clrs_t).export(f"./misc/custom_{scene_name}_transient.ply")
            trimesh.PointCloud(vertices=pts_r, colors=clrs_r).export(f"./misc/custom_{scene_name}_rigid.ply")

            trimesh.PointCloud(
                vertices=torch.cat([pts_t, pts_r], dim=0), 
                colors=torch.cat([clrs_t_red, clrs_r_blue], dim=0)
            ).export(f"./misc/custom_{scene_name}_transient_rigid_color_coded.ply")
        except:
            continue