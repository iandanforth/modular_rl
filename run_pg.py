#!/usr/bin/env python

"""
This script runs a policy gradient algorithm
"""

# Standard Library
import argparse
import sys
import cPickle
import shutil
import os
import logging

# Third Party
import gym
import numpy as np
from gym.envs import make
from tabulate import tabulate

# Local
from modular_rl.core import (VIDEO_NEVER,
                             get_agent_cls,
                             animate_rollout,
                             run_policy_gradient_algorithm)
from modular_rl.misc_utils import (update_argument_parser,
                                   prepare_h5_file,
                                   GENERAL_OPTIONS)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add defaults into parser
    update_argument_parser(parser, GENERAL_OPTIONS)

    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--plot", action="store_true")

    # Pull out computed args
    args, _ = parser.parse_known_args(
        [arg for arg in sys.argv[1:] if arg not in ('-h', '--help')]
    )

    # Initialize the Gym env
    env = make(args.env)
    env_spec = env.spec

    # Create our output directory
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir):
        shutil.rmtree(mondir)
    os.mkdir(mondir)

    # Start our environment monitor
    video_callable = None
    if not args.video:
        video_callable = VIDEO_NEVER
    env.monitor.start(mondir, video_callable=video_callable)

    # Get constructor for the agent class requested in args
    agent_ctor = get_agent_cls(args.agent)

    # Update our parser with options defined in this agent class
    update_argument_parser(parser, agent_ctor.options)

    # Get final computed args
    args = parser.parse_args()

    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit

    # Use our computed args as our config dict
    cfg = args.__dict__

    # Set our random seed for deterministic results
    np.random.seed(args.seed)

    # Instantiate the requested agent
    agent = agent_ctor(env.observation_space, env.action_space, cfg)

    # Prepare results/snapshot file if requested. Default is False.
    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)

    gym.logger.setLevel(logging.WARN)

    # Define the callback to be passed to the PG runner.
    # Primarily for debugging/logging/progress.
    COUNTER = 0
    def callback(stats):
        global COUNTER
        COUNTER += 1
        # Print stats
        print "*********** Iteration %i ****************" % COUNTER

        # Remove stats that can't be easilyt displayed in a table
        iteration_stats = [(k, v) for k, v in stats.items()
                           if np.asarray(v).size == 1]
        # Pretty print the stats array
        print tabulate(iteration_stats)
        # Store to hdf5
        if args.use_hdf:
            for (stat, val) in stats.items():
                if np.asarray(val).ndim == 0:
                    diagnostics[stat].append(val)
                else:
                    assert val.ndim == 1
                    diagnostics[stat].extend(val)
            snapshot_time = (
                (COUNTER % args.snapshot_every == 0) or (COUNTER == args.n_iter)
            )
            if args.snapshot_every and snapshot_time:
                hdf['/agent_snapshots/%0.4i' % COUNTER] = \
                np.array(cPickle.dumps(agent, -1))
        # Plot (render each trial)
        if args.plot:
            animate_rollout(env, agent, min(500, args.timestep_limit))

    # Where the magic happens!
    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg=cfg)

    if args.use_hdf:
        hdf['env_id'] = env_spec.id
        try:
            hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception:
            print "failed to pickle env"

    # Clean up our Gym environment
    env.monitor.close()
