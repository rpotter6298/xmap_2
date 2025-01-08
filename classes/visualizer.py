import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

class Visualizer:
    """
    A static class for various data visualization methods.
    """
    @staticmethod
    def plot_clustered_heatmap(
        corr_matrix, pval_matrix, title="Clustered Correlation Heatmap",
        significance_level=0.05, cmap="coolwarm", figsize=(28, 12), 
        method='average', metric='correlation', cbar_pos=(0.02, 0.75, 0.02, 0.2), 
        dendrogram_ratio=(0.1, 0.1), save_path=None, text_x=0.92, text_y=0.1, clean_X_names=False, 
        clean_Y_names=False, column_order=None, cluster_rows=True, asterisk_size=16
    ):
        """
        Generates a clustered heatmap with annotated p-value significance.

        Args:
            corr_matrix (pd.DataFrame): Correlation matrix.
            pval_matrix (pd.DataFrame): Corresponding p-value matrix.
            title (str): Title of the heatmap.
            significance_level (float): Threshold for significance.
            cmap (str): Colormap for the heatmap.
            figsize (tuple): Figure size.
            method (str): Linkage method for clustering.
            metric (str): Distance metric for clustering.
            cbar_pos (tuple): Position of the colorbar.
            dendrogram_ratio (tuple): Ratios for dendrogram size.
            save_path (str, optional): Path to save the plot.
            text_x (float): X-coordinate for annotation text.
            text_y (float): Y-coordinate for annotation text.
        """

        # Apply custom column order if provided
        if column_order is not None:
            # Ensure provided order matches existing columns
            missing_columns = [col for col in column_order if col not in corr_matrix.columns]
            if missing_columns:
                raise ValueError(f"Columns in 'column_order' not found in corr_matrix: {missing_columns}")
            
            # Reorder correlation and p-value matrices based on the specified column order
            corr_matrix = corr_matrix.loc[:, column_order]
            pval_matrix = pval_matrix.loc[:, column_order]

            # Disable column clustering since column order is manually specified
            cluster_columns = False
        else:
            cluster_columns = True


        #Helper function to clean names
        def clean_names(names):
            cleaned= []
            counts = {}
            for name in names:
                base_name = name.split("_")[0]
                if base_name not in counts:
                    counts[base_name] = 1
                    cleaned.append(base_name)
                else:
                    cleaned.append(f"{base_name}_{counts[base_name]}")
                    counts[base_name] += 1
            return cleaned

        # Clean names if flags are set
        x_labels = corr_matrix.columns.tolist()
        y_labels = corr_matrix.index.tolist()

        if clean_X_names:
            x_labels = clean_names(x_labels)
        if clean_Y_names:
            y_labels = clean_names(y_labels)

        # Generate clustering linkage for columns and rows
        if cluster_columns:
            col_linkage = linkage(corr_matrix.T, method=method, metric=metric)
        else:
            col_linkage = None  # No clustering for columns if column_order is specified
        
        if cluster_rows:
            row_linkage = linkage(corr_matrix, method=method, metric=metric)
        else:
            row_linkage = None  # No clustering for rows if not requested

        # Create the clustered heatmap
        g = sns.clustermap(
            corr_matrix,
            cmap=cmap,
            linewidths=0.5,
            figsize=figsize,
            row_cluster=cluster_rows,
            col_cluster=cluster_columns,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            annot=False,
            xticklabels=x_labels,
            yticklabels=y_labels,
            #square=False,  # Fixed: Seaborn doesn't handle this in clustermap
            cbar_pos=cbar_pos,
            dendrogram_ratio=dendrogram_ratio,
        )

        # Adjust font sizes for x and y tick labels
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=14)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=14)
        # Hide the row dendrogram but keep clustering
        g.ax_row_dendrogram.set_visible(False)

        plt.subplots_adjust(right=0.9, bottom=0.15, top=0.9)  # Adjust top margin (default is 1.0)
        
        g.figure.suptitle(title, fontsize=16, y=.95)
        g.figure.text(text_x, text_y, f'* = p < {significance_level:.2f}', fontsize=16, color='black', ha='center')

        # Correctly align asterisks based on reordered data
        if cluster_rows:
            row_order = g.dendrogram_row.reordered_ind
        else:
            row_order = range(len(y_labels))
        if cluster_columns:
            col_order = g.dendrogram_col.reordered_ind
        else:
            col_order = range(len(x_labels))        
        g.ax_cbar.set_position(cbar_pos)

        for i, row_idx in enumerate(row_order):
            for j, col_idx in enumerate(col_order):
                pval = pval_matrix.iloc[row_idx, col_idx]
                if pval < significance_level:
                    g.ax_heatmap.text(j + 0.5, i + 0.5, '*', color='black', 
                                      ha='center', va='center', fontsize=asterisk_size, weight='bold')

        # Display or save the plot
        if save_path:
            plt.savefig(save_path, format="jpeg", dpi=300)
            print(f"Plot saved as '{save_path}'")
        else:
            plt.show()
    @staticmethod
    def plot_dendrogram(data, method='average', metric='correlation', figsize=(12, 8), save_path=None):
        """
        Plots the original dendrogram without modifying the linkage heights.

        Args:
            data (ndarray or DataFrame): Data for clustering.
            method (str): Linkage method for clustering.
            metric (str): Distance metric for clustering.
            figsize (tuple): Figure size for the plot.
        """
        # Generate linkage matrix
        linkage_matrix = linkage(data, method=method, metric=metric)

        # Plot the dendrogram
        plt.figure(figsize=figsize)
        dendrogram(linkage_matrix, labels=data.columns if hasattr(data, "columns") else None)

        # Adjust the title and axis labels
        plt.title("Protein Clusters Dendrogram (Original)")
        plt.ylabel("Distance")
        plt.xlabel("Proteins")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format="jpeg", dpi=300)
            print(f"Plot saved as '{save_path}'")
        else:
            plt.show()

                    
    def plot_dendrogram_with_height_transform(
        data, 
        method='average', 
        metric='correlation', 
        transform_factor=100, 
        clean_labels=False,
        title_fontsize=18,
        label_fontsize=16,
        tick_fontsize=14,
        save_path=None
    ):
        """
        Plots a dendrogram with transformed linkage heights using sqrt(transform_factor * x).
        
        Args:
            data (ndarray or DataFrame): Data for clustering.
            method (str): Linkage method for clustering.
            metric (str): Distance metric for clustering.
            transform_factor (float): Factor to scale distances before applying sqrt.
            clean_labels (bool): Whether to clean column names to remove trailing numbers
                                and replace duplicates with incremented suffixes.
            title_fontsize (int): Font size for the plot title.
            label_fontsize (int): Font size for the axis labels.
            tick_fontsize (int): Font size for the tick labels.
        """
        
        # Helper function to clean names
        def clean_names(names):
            cleaned = []
            counts = {}
            for name in names:
                base_name = name.split("_")[0]
                if base_name not in counts:
                    counts[base_name] = 1
                    cleaned.append(base_name)
                else:
                    cleaned.append(f"{base_name}_{counts[base_name]}")
                    counts[base_name] += 1
            return cleaned

        # Compute original linkage matrix
        linkage_matrix = linkage(data, method=method, metric=metric)

        # Transform the heights in the linkage matrix
        transformed_linkage_matrix = linkage_matrix.copy()
        transformed_linkage_matrix[:, 2] = np.sqrt(transform_factor * transformed_linkage_matrix[:, 2])

        # Determine labels
        labels = data.columns.tolist() if hasattr(data, "columns") else None
        if clean_labels and labels is not None:
            labels = clean_names(labels)

        # Plot the dendrogram using the transformed heights
        plt.figure(figsize=(14, 10))  # Increased figure size for better readability
        dendro = dendrogram(transformed_linkage_matrix, labels=labels, leaf_rotation=90, 
                        leaf_font_size=tick_fontsize)

        # Adjust the title and axis labels with increased font sizes
        ylabel = f"Transformed Distance (sqrt({transform_factor} * x))"
        plt.title("Protein Clusters Dendrogram (Transformed Heights)", fontsize=title_fontsize)
        plt.ylabel(ylabel, fontsize=label_fontsize)
        plt.xlabel("Proteins", fontsize=label_fontsize)

        # Customize tick label sizes
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format="jpeg", dpi=300)
            print(f"Plot saved as '{save_path}'")
        else:
            plt.show()