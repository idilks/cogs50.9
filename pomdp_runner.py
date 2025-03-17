import os

from models import RockSampleModel, Model
from solvers import POMCP, PBVI
from parsers import PomdpParser, GraphViz
from logger import Logger as log
import matplotlib.pyplot as plt
import numpy as np


class PomdpRunner:

    def __init__(self, params):
        self.params = params
        if params.logfile is not None:
            log.new(params.logfile)

    def create_model(self, env_configs):
        """
        Builder method for creating model (i,e, agent's environment) instance
        :param env_configs: the complete encapsulation of environment's dynamics
        :return: concrete model
        """
        MODELS = {
            'RockSample': RockSampleModel,
        }
        return MODELS.get(env_configs['model_name'], Model)(env_configs)

    def create_solver(self, algo, model):
        """
        Builder method for creating solver instance
        :param algo: algorithm name
        :param model: model instance, e.g, TigerModel or RockSampleModel
        :return: concrete solver
        """
        SOLVERS = {
            'pbvi': PBVI,
            'pomcp': POMCP,
        }
        return SOLVERS.get(algo)(model)

    def snapshot_tree(self, visualiser, tree, filename):
        visualiser.update(tree.root)
        visualiser.render('./dev/snapshots/{}'.format(filename))  # TODO: parametrise the dev folder path

    def run(self, algo, T, **kwargs):
        visualiser = GraphViz(description='tmp')
        params, pomdp = self.params, None
        total_rewards, budget = 0, params.budget

        log.info('~~~ initialising ~~~')
        with PomdpParser(params.env_config) as ctx:
            # creates model and solver
            model = self.create_model(ctx.copy_env())
            pomdp = self.create_solver(algo, model)

            # supply additional algo params
            belief = ctx.random_beliefs() if params.random_prior else ctx.generate_beliefs()

            if algo == 'pbvi':
                belief_points = ctx.generate_belief_points(kwargs['stepsize'])
                pomdp.add_configs(belief_points)
            elif algo == 'pomcp':
                pomdp.add_configs(budget, belief, **kwargs)

        # have fun!
        log.info('''
        ++++++++++++++++++++++
            Starting State:  {}
            Starting Budget:  {}
            Init Belief: {}
            Time Horizon: {}
            Max Rounds: {}
        ++++++++++++++++++++++'''.format(model.curr_state, budget, belief, T, params.max_play))

        ## lists for tracking results
        rounds = []
        rewards = []
        assets = []
        beliefs = []
        actions = []
        observations = []
        states = []
        
        
        for i in range(params.max_play):
            # plan, take action and receive environment feedbacks
            pomdp.solve(T)
            action = pomdp.get_action(belief)
            new_state, obs, reward, cost = pomdp.take_action(action)

            if params.snapshot and isinstance(pomdp, POMCP):
                # takes snapshot of belief tree before it gets updated
                self.snapshot_tree(visualiser, pomdp.tree, '{}.gv'.format(i))
            
            # update states
            belief = pomdp.update_belief(belief, action, obs)
            total_rewards += reward
            budget -= cost
            
            
            # store results
            rounds.append(i + 1)
            rewards.append(total_rewards)
            assets.append(budget)
            beliefs.append(belief)  # storing full belief state
            actions.append(action)
            observations.append(obs)
            states.append(new_state)

            # print ino
            log.info('\n'.join([
              'Taking action: {}'.format(action),
              'Observation: {}'.format(obs),
              'Reward: {}'.format(reward),
              'Budget: {}'.format(budget),
              'New state: {}'.format(new_state),
              'New Belief: {}'.format(belief),
              '=' * 20
            ]))

            if budget <= 0:
                log.info('Budget spent.')
                break
        
        
        self.plot_results_action(rounds, assets, actions)

        log.info('{} actions taken. Total reward = {}'.format(i + 1, total_rewards))
        return pomdp


    def plot_results_action(self, rounds, assets, actions):
        unique_actions = list(set(actions))  # get unique actions
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_actions)))  # assign colors
        action_color_map = dict(zip(unique_actions, colors))  # map actions to colors
        # Budget Decrease Over Time with Actions
        plt.figure(figsize=(10, 5))
        time = np.arange(len(assets))
        
        plt.plot(time, assets, marker="s", linestyle="--", color="red", label="Budget Remaining")

        for action, color in action_color_map.items():
            action_indices = [i for i, a in enumerate(actions) if a == action]
            action_budgets = [assets[i] for i in action_indices]
            plt.scatter(action_indices, action_budgets, color=color, label=f"Action: {action}", s=50)

            plt.xlabel("Time")
            plt.ylabel("Budget")
            plt.title("Budget Decrease Over Time with Actions")
            plt.legend()
            plt.grid()
            # plt.savefig(os.path.join(os.getcwd(), "results", "budget_decrease_over_time_with_actions.png"))
            plt.show()
                
