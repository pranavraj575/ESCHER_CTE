if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    all_trials = ['default',  # default ESCHER
                  'ESCHER_CTE',  # ESCHER CTE
                  'true_NE',  # ESCHER given a mixed version of the true NE to start with
                  'true_NE_warm',  # true NE + warm starting
                  ]
    # TODO: implement warm starts
    parser.add_argument('--game', required=True, help='name of pyspiel game', action='store')
    parser.add_argument('--trial', required=False, default='default', help='trial to use', choices=all_trials)
    parser.add_argument('--plot', required=False, action='store_true', help='only plot past results')
    parser.add_argument('--recollect', required=False, action='store_true', help='collect results again')
    parser.add_argument('--p',
                        required=False,
                        default=0.5,
                        help='if mixing reference policy, how much should the mixed (i.e. uniform) policy be included',
                        type=float,
                        )

    args = parser.parse_args()
    from ESCHER import ESCHERSolver, PolicyMixer
    import pyspiel
    import numpy as np
    import matplotlib.pyplot as plt, os

    train_device = 'cpu'
    game_name = args.game

    TRIAL = args.trial
    recollect = args.recollect
    DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(DIR, 'tmp', 'results', game_name, TRIAL)
    os.makedirs(save_path, exist_ok=True)

    game = pyspiel.load_game(game_name)

    convs_path = save_path + '_convs.npy'
    if not args.plot:
        print('saving convs to', convs_path)

    ref_policy = None
    if TRIAL == 'true_NE' or TRIAL == 'true_NE_warm':
        solver = pyspiel.CFRPlusSolver(game)
        print('solving game directly using pyspiel\'s CFR')
        for i in range(1420):
            if not i%100:
                average_policy = solver.average_policy()
                exploit = pyspiel.exploitability(game, average_policy)
                print(f"iteration {i}; Exploitability: {exploit}")
            solver.evaluate_and_update_policy()
        average_policy = solver.average_policy()
        ref_policy = PolicyMixer(average_policy, game=game, uniform_p=args.p)

    iters = 100
    num_traversals = 500
    num_val_fn_traversals = 500
    regret_train_steps = 200
    val_train_steps = 200
    policy_net_train_steps = 1000
    batch_size_regret = 256
    batch_size_val = 256

    policy_network_layers = (128, 64)
    regret_network_layers = (128, 64)
    value_network_layers = (128, 64)

    check_exploitability_every = 10

    if not args.plot:
        if recollect or not os.path.exists(convs_path):
            deep_cfr_solver = ESCHERSolver(
                game,
                policy_network_layers=policy_network_layers,
                regret_network_layers=regret_network_layers,
                value_network_layers=value_network_layers,
                num_traversals=int(num_traversals),
                num_iterations=iters,
                check_exploitability_every=check_exploitability_every,
                compute_exploitability=True,
                regret_network_train_steps=regret_train_steps,
                policy_network_train_steps=policy_net_train_steps,
                batch_size_regret=batch_size_regret,
                value_network_train_steps=val_train_steps,
                batch_size_value=batch_size_val,
                train_device=train_device,
                reference_policy=ref_policy,
            )
            regret, pol_loss, convs, nodes = deep_cfr_solver.solve(save_path_convs=save_path)
    if args.plot:
        for trial in all_trials:
            convs_path = os.path.join(DIR, 'tmp', 'results', game_name, trial) + '_convs.npy'
            if os.path.exists(convs_path):
                print('loading convs from', convs_path)
                convs = np.load(convs_path)
                print(convs)
                plt.plot(convs, label=trial)
        plt.legend()
        plt.show()
        plt.close()
