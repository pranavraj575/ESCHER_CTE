from ESCHER import ESCHERSolver, PolicyMixer
import pyspiel
import numpy as np

if __name__ == "__main__":
    import matplotlib.pyplot as plt, os

    train_device = 'cpu'
    game_name = "leduc_poker"

    TRIAL = 'default'  # default ESCHER
    # TRIAL='ESCHER_CTE' # ESCHER CTE
    TRIAL = 'true_NE'  # ESCHER given a mixed version of the true NE to start with
    only_plot = False
    DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(DIR, 'tmp', 'results', game_name, TRIAL)
    os.makedirs(save_path, exist_ok=True)

    game = pyspiel.load_game(game_name)
    all_trials = ['default', 'ESCHER_CTE', 'true_NE']
    convs_path = save_path + '_convs.npy'
    if not only_plot:
        print('saving convs to', convs_path)

    ref_policy = None
    if TRIAL == 'true_NE':
        solver = pyspiel.CFRPlusSolver(game)
        print('solving game directly using pyspiel\'s CFR')
        for i in range(1420):
            if not i%100:
                average_policy = solver.average_policy()
                exploit = pyspiel.exploitability(game, average_policy)
                print(f"iteration {i}; Exploitability: {exploit}")
            solver.evaluate_and_update_policy()
        average_policy = solver.average_policy()
        ref_policy = PolicyMixer(average_policy, game=game, uniform_p=1)

    iters = 100
    num_traversals = 500
    num_val_fn_traversals = 500
    regret_train_steps = 200
    val_train_steps = 200
    policy_net_train_steps = 1000
    batch_size_regret = 256
    batch_size_val = 256
    if not only_plot:
        deep_cfr_solver = ESCHERSolver(
            game,
            num_traversals=int(num_traversals),
            num_iterations=iters,
            check_exploitability_every=10,
            compute_exploitability=True,
            regret_network_train_steps=regret_train_steps,
            policy_network_train_steps=policy_net_train_steps,
            batch_size_regret=batch_size_regret,
            value_network_train_steps=val_train_steps,
            batch_size_value=batch_size_val,
            train_device=train_device,
            reference_policy=ref_policy,
        )
        if not os.path.exists(convs_path):
            regret, pol_loss, convs, nodes = deep_cfr_solver.solve(save_path_convs=save_path)
            plt.plot(convs)
            plt.show()
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
