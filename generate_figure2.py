"""
generate_figure2.py
====================
Regenerates Figure 2 (Zero-Day Impact) with correct
dual Y-axis design.

Left Y-axis:  Number of Zero-Day Exploits (count)
Right Y-axis: Estimated Cybercrime Cost (USD Trillion)
X-axis:       Year (2023-2028)

This addresses Review Point 5.4:
  "Figure 2 shows exploits count and cost on the same
   horizontal axis — these cannot share an axis."

Run from project root:
    python generate_figure2.py

Output:
    Zero-Day-Impact.png  (replace existing file)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

#    Data (from Statista / Google Project Zero 2023-2028)                  
years = [2023, 2024, 2025, 2026, 2027, 2028]

# Zero-day exploits: observed + projected counts
# Source: Google Project Zero / industry reports
exploits = [97, 106, 115, 126, 138, 151]

# Cybercrime costs in USD Trillion
# Source: Statista cybercrime cost projections
costs = [8.0, 9.5, 10.5, 11.5, 13.0, 15.0]

PROJECT_ROOT = r"C:/MLProject/zero_day_project"

def run():
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    #    Left Y-axis: Exploits                                              
    color_exploits = "#1B5EA6"
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Number of Zero-Day Exploits",
                   color=color_exploits, fontsize=11)
    bars = ax1.bar(
        [y - 0.2 for y in years], exploits,
        width=0.35,
        color=color_exploits,
        alpha=0.85,
        label="Zero-Day Exploits"
    )
    ax1.tick_params(axis="y", labelcolor=color_exploits)
    ax1.set_ylim(0, 200)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

    # Add value labels on bars
    for bar, val in zip(bars, exploits):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(val),
            ha="center", va="bottom",
            fontsize=9, color=color_exploits, fontweight="bold"
        )

    #    Right Y-axis: Cost                                                 
    ax2 = ax1.twinx()
    color_cost = "#C0392B"
    ax2.set_ylabel("Estimated Cybercrime Cost (USD Trillion)",
                   color=color_cost, fontsize=11)
    line = ax2.plot(
        years, costs,
        color=color_cost,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Cybercrime Cost (USD Trillion)"
    )
    ax2.tick_params(axis="y", labelcolor=color_cost)
    ax2.set_ylim(0, 20)
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"${x:.0f}T")
    )

    # Add value labels on line
    for x, y in zip(years, costs):
        ax2.annotate(
            f"${y}T",
            xy=(x, y),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=9,
            color=color_cost,
            fontweight="bold"
        )

    #    Legend                                                             
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        fontsize=9,
        framealpha=0.9
    )

    #    X-axis                                                             
    ax1.set_xticks(years)
    ax1.set_xticklabels(years, fontsize=10)

    # Shade projected years
    ax1.axvspan(2025.5, 2028.5, alpha=0.06, color="gray")
    ax1.text(
        2027, 180, "Projected",
        ha="center", fontsize=9,
        color="gray", style="italic"
    )

    #    Title and layout                                                   
    ax1.set_title(
        "Zero-Day Exploit Activity and Cybercrime Cost Trends (2023–2028)",
        fontsize=12, fontweight="bold", pad=12
    )

    plt.tight_layout()

    #    Save                                                               
    # Save to project root (where LaTeX expects it)
    out_path = os.path.join(PROJECT_ROOT, "Zero-Day-Impact.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Saved -> {out_path}")

    # Also save to outputs folder
    out_local = "/mnt/user-data/outputs/Zero-Day-Impact.png" \
        if os.path.exists("/mnt/user-data/outputs") \
        else "Zero-Day-Impact.png"
    plt.figure()  # dummy to avoid saving closed fig
    # Re-run the plot for local save
    fig2, ax1b = plt.subplots(figsize=(8, 4.5))
    color_exploits = "#1B5EA6"
    color_cost     = "#C0392B"

    ax1b.set_xlabel("Year", fontsize=12)
    ax1b.set_ylabel("Number of Zero-Day Exploits",
                    color=color_exploits, fontsize=11)
    bars2 = ax1b.bar(
        [y - 0.2 for y in years], exploits, width=0.35,
        color=color_exploits, alpha=0.85,
        label="Zero-Day Exploits"
    )
    ax1b.tick_params(axis="y", labelcolor=color_exploits)
    ax1b.set_ylim(0, 200)
    ax1b.yaxis.set_major_locator(ticker.MultipleLocator(50))
    for bar, val in zip(bars2, exploits):
        ax1b.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height()+3, str(val),
                  ha="center", va="bottom",
                  fontsize=9, color=color_exploits,
                  fontweight="bold")

    ax2b = ax1b.twinx()
    ax2b.set_ylabel("Estimated Cybercrime Cost (USD Trillion)",
                    color=color_cost, fontsize=11)
    ax2b.plot(years, costs, color=color_cost, marker="o",
              linewidth=2.5, markersize=7,
              label="Cybercrime Cost (USD Trillion)")
    ax2b.tick_params(axis="y", labelcolor=color_cost)
    ax2b.set_ylim(0, 20)
    ax2b.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"${x:.0f}T")
    )
    for x, y in zip(years, costs):
        ax2b.annotate(f"${y}T", xy=(x, y),
                      xytext=(5, 8),
                      textcoords="offset points",
                      fontsize=9, color=color_cost,
                      fontweight="bold")

    h1, l1 = ax1b.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax1b.legend(h1+h2, l1+l2, loc="upper left",
                fontsize=9, framealpha=0.9)
    ax1b.set_xticks(years)
    ax1b.set_xticklabels(years, fontsize=10)
    ax1b.axvspan(2025.5, 2028.5, alpha=0.06, color="gray")
    ax1b.text(2027, 180, "Projected", ha="center",
              fontsize=9, color="gray", style="italic")
    ax1b.set_title(
        "Zero-Day Exploit Activity and Cybercrime Cost "
        "Trends (2023–2028)",
        fontsize=12, fontweight="bold", pad=12
    )
    plt.tight_layout()
    plt.savefig(out_local, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Also saved -> {out_local}")
    print("\nDone. Copy Zero-Day-Impact.png to your LaTeX folder.")


if __name__ == "__main__":
    run()
