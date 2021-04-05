#!/usr/bin/env python3

import time
import numpy as np

from phonebot.core.common.util import get_time
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.math.utils import anorm, alerp
from phonebot.core.common.math.transform import Rotation
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.controls.agents.point_tracker_agent.point_tracker_agent import PointTrackerAgent
from phonebot.core.controls.agents.trajectory_agent import TrajectoryAgentGraph
from phonebot.core.controls.agents.base_agent.base_agent import BaseAgent
from phonebot.core.frame_graph.graph_utils import (
    initialize_graph_nominal, update_passive_joints, get_joint_edges)


from phonebot.core.kinematics.workspace import get_workspace, max_rect
from phonebot.core.controls.trajectory.trajectory_utils import get_elliptical_trajectory


class JointAngleAgent(BaseAgent):
    def __init__(self, config: PhonebotSettings):
        self.config = config
        self.target = 0.0
        self.num_joints = len(self.config.active_joint_names)

    def set_target(self, angle: float):
        self.target = angle

    def __call__(self, state, stamp):
        return np.full(self.num_joints, self.target, dtype=np.float32)


class OpenloopJointGraphUpdater(object):
    """
    Update the graph in an open-loop manner
    solely based on the servo angles.
    """

    def __init__(self, graph, config):
        self.graph = graph
        self.config = config
        # NOTE(yycho0108): joint_edges == active joint edges (x8)
        self.joint_edges = get_joint_edges(self.graph, self.config)
        self.commands = np.zeros(len(self.joint_edges))

    def update_commands(self, commands: np.ndarray):
        """ Save current commands. """
        # TODO(yycho0108): Store timestamp and apply timeout.
        self.commands = commands

    def update_graph(self, stamp: float):
        # Alpha factor for interpolation.
        # TODO(yycho0108): Add interpolation constant to config.
        # TODO(yycho0108): Maybe the alpha factor should depend on time.
        alpha = 1.0

        # Update a linearly interpolated joint value based on commands.
        # print(self.commands)
        for joint_edge, joint_command in zip(self.joint_edges, self.commands):
            joint_edge.update(stamp, anorm(
                alerp(joint_edge.angle, joint_command, alpha)))
        # Update passive joints based on the above active-joint updates.
        # print('<update_passive_joints>')
        update_passive_joints(self.graph, stamp, self.config)
        # print('</update_passive_joints>')


def get_trajectories(config: PhonebotSettings = PhonebotSettings()):
    # Get maximal ellipse bounds.
    if config.use_cached_rect:
        ws_rect = config.max_rect
    else:
        workspace = get_workspace(0.0, config, return_poly=True)
        ws_rect = max_rect(workspace, 4096)
    (x0, y0), (x1, y1) = ws_rect
    ws_rect = (0.5 * (x0+x1), 0.5 * (y0 + y1),
               abs(x1-x0), abs(y1-y0))

    # NOTE(yycho0108): 0.8 still has transient issues, 0.7 seems to be stable
    # TODO(yycho0108): Maybe the lag
    trajectories = {
        'FL': get_elliptical_trajectory(ws_rect, 2.0, -np.pi/2, False, 0.7),
        'FR': get_elliptical_trajectory(ws_rect, 2.0, np.pi/2, True, 0.7),
        'HL': get_elliptical_trajectory(ws_rect, 2.0, np.pi/2, False, 0.7),
        'HR': get_elliptical_trajectory(ws_rect, 2.0, -np.pi/2, True, 0.7)
    }
    return trajectories


class Phonebot(object):

    def __init__(self, config: PhonebotSettings = PhonebotSettings()):
        self.config = config
        self.graph = PhonebotGraph(self.config)
        self.initialize()
        self.joint_edges = get_joint_edges(self.graph, self.config)

        # NOTE(yycho0108): hardcoded assumption to start from nominal hip angle.
        self.commands = np.full(len(self.joint_edges),
                                self.config.nominal_hip_angle)
        self.state_estimator = OpenloopJointGraphUpdater(
            self.graph, self.config)

        # Worst way to make a state machine ever
        self.trajectories = get_trajectories(config)
        self.agent_map = {
            'idle': JointAngleAgent(self.config),
            'stand_up': JointAngleAgent(self.config),
            # 'idle': TrajectoryAgentGraph(self.graph, 2.0, self.config, trajectories=self.trajectories),
            # 'stand_up': TrajectoryAgentGraph(self.graph, 2.0, self.config, trajectories=self.trajectories),
            'walk': TrajectoryAgentGraph(self.graph, 2.0, self.config, trajectories=self.trajectories),
            'face_follow': PointTrackerAgent(self.graph, self.config),
            'sit_down': JointAngleAgent(self.config),
        }
        self.agent_map['idle'].set_target(np.deg2rad(10))
        self.agent_map['stand_up'].set_target(self.config.nominal_hip_angle)
        self.agent_map['sit_down'].set_target(np.deg2rad(10))
        self.state = 'idle'

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def initialize(self):
        stamp = get_time()
        initialize_graph_nominal(self.graph, stamp, self.config)
        update_passive_joints(self.graph, stamp, self.config)

    def update_angle(self, roll: float, pitch: float, alpha: float = 0.5):
        """ Set current orientation. """
        if self.state == 'face_follow':
            self.agent_map[self.state].update_orientation(roll, pitch, alpha)

    def set_target(self, roll: float, pitch: float, relative: bool = False):
        """ Set orientation target. """
        # TODO(yycho0108): Verify if pitch,roll here is in the right order.
        if self.state == 'face_follow':
            self.agent_map[self.state].update_target(roll, pitch, relative)

    def get_commands(self, stamp: float):
        # TODO(yycho0108): __call__ for agents is maybe not the right choice.
        return self.agent_map[self.state](None, stamp)

    def step(self, stamp: float):
        """
        Update the current state and return computed joint commands.
        """
        # print('<source>')
        for prefix in self.config.order:
            body_from_foot = self.graph.get_transform(
                '{}_foot_a'.format(prefix), 'body', stamp)
            # print(body_from_foot.position)
        # print('</source>')

        # print('<get_commands>')

        # Invoke the (open loop) state estimator.
        # print('stamp = {}'.format(stamp))
        # print('<get_commands>')
        self.commands = self.get_commands(stamp)
        # print('</get_commands>')

        # print('<trajectory>')
        # for prefix in self.config.order:
        #     print('{} : {}'.format(
        #         prefix, self.trajectories[prefix].evaluate(stamp)))
        # print('</trajectory>')
        self.state_estimator.update_commands(self.commands)
        self.state_estimator.update_graph(stamp)
        return self.commands


def main():
    robot = Phonebot()
    start = time.time()
    robot.initialize()
    while True:
        now = time.time()
        if (now - start) >= 5.0:
            break
        robot.update_angle(0.1, 0.1)
        robot.set_target(0.2, 0.2, relative=True)
        commands = robot.step(now)
        # print(commands)


if __name__ == '__main__':
    main()
