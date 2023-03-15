"""
Helper file for generating an environment rollout
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d


def generate_tag_env_rollout_animation(
    trainer,
    fps=50,
    tagger_color="#C843C3",
    runner_color="#245EB6",
    runner_not_in_game_color="#666666",
    fig_width=6,
    fig_height=6,
):
    assert trainer is not None

    # Fetch episode states
    episode_states = trainer.fetch_episode_states(
        ["loc_x", "loc_y", "still_in_the_game"]
    )
    assert isinstance(episode_states, dict)
    env = trainer.cuda_envs.env

    fig, ax = plt.subplots(
        1, 1, figsize=(fig_width, fig_height)
    )  # , constrained_layout=True
    ax.remove()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Bounds
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.01, 0.01)

    # Surface
    corner_points = [(0, 0), (0, 1), (1, 1), (1, 0)]

    poly = Polygon(corner_points, color=(0.1, 0.2, 0.5, 0.15))
    ax.add_patch(poly)
    art3d.pathpatch_2d_to_3d(poly, z=0, zdir="z")

    # "Hide" side panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Hide grid agents_drawing
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Hide axes
    ax.set_axis_off()
    # Set camera
    ax.set_box_aspect((40, -55, 10))

    # Try to reduce whitespace
    fig.subplots_adjust(left=0, right=1, bottom=-0.2, top=1)

    # Init agents_drawing
    agents_drawing = [None for _ in range(env.num_agents)]

    # Plot init data
    for idx in range(env.num_agents):
        if idx in env.predators:
            agents_drawing[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx] / env.stage_size,
                episode_states["loc_y"][:1, idx] / env.stage_size,
                0,
                color=tagger_color,
                marker="o",
                markersize=10,
            )[0]
        else:  # preys
            agents_drawing[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx] / env.stage_size,
                episode_states["loc_y"][:1, idx] / env.stage_size,
                [0],
                color=runner_color,
                marker="o",
                markersize=5,
            )[0]

    init_num_preys = env.num_agents - env.num_predators

    # Label the plot
    def _get_label(timestep, n_preys_alive, init_n_preys):
        line1 = "Continuous Tag\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        frac_preys_alive = n_preys_alive / init_n_preys
        pct_preys_alive = f"{n_preys_alive:4} ({frac_preys_alive * 100:.0f}%)"
        line3 = "preys Left:".ljust(14) + pct_preys_alive
        return line1 + line2 + line3

    label = ax.text(
        0,
        0,
        0.02,
        _get_label(0, init_num_preys, init_num_preys).lower(),
    )

    label.set_fontsize(14)
    label.set_fontweight("normal")
    label.set_color("#666666")

    # Animate
    def animate(i):
        for idx, drawing in enumerate(agents_drawing):
            # Update drawing
            drawing.set_data_3d(
                episode_states["loc_x"][i : i + 1, idx] / env.stage_size,
                episode_states["loc_y"][i : i + 1, idx] / env.stage_size,
                np.zeros(1),
            )

            still_in_game = episode_states["still_in_the_game"][i, idx]

            if still_in_game:
                pass
            else:
                drawing.set_color(runner_not_in_game_color)
                drawing.set_marker("")

        n_preys_alive = episode_states["still_in_the_game"][i].sum() - env.num_predators
        label.set_text(_get_label(i, n_preys_alive, init_num_preys).lower())

    ani = animation.FuncAnimation(
        fig, animate, np.arange(0, env.episode_length + 1), interval=1000.0 / fps
    )
    plt.close()

    return ani