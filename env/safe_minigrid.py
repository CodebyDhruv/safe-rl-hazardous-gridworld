from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.core.mission import MissionSpace
from env.hazard import Hazard

class SafeMiniGridEnv(MiniGridEnv):
    def __init__(
        self,
        size=16,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps=None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.hazards = {
    (5, 3), (5, 4), (5, 5), (5, 6),
    (8, 6), (9, 6),
    (7, 9), (8, 9),
    (12, 13), (13, 13), (13, 12),
    (10, 10), (11, 10), (12, 10),
    (3, 11)
}
        mission_space = MissionSpace(
            mission_func=lambda: "Reach the goal while avoiding hazards"
        )

        if max_steps is None:
            max_steps = 4 * size * size

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True, 
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Goal(), width - 2, height - 2)
        for hx, hy in self.hazards:
            self.grid.set(hx, hy, Hazard())
            
        if self.agent_start_pos:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Reach the goal while avoiding hazards"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward -= 0.01
        violation = 0
        if tuple(self.agent_pos) in self.hazards:
            violation = 1

        info["violation"] = violation
        info["agent_pos"] = tuple(self.agent_pos)

        return obs, reward, terminated, truncated, info

